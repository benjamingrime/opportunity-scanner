"""The scan: a workforce (occupations and how many people) -> where AI could help.

Plain code, no LLM. Each occupation's tasks are split by the exposure index into
  E1 "buy and train": an off-the-shelf AI assistant could halve the time,
  E2 "build":         it needs software or an agent connected to the organisation's systems,
  E0 "leave":         AI can't halve the time; people stay central.
Results are shares of working time, weighted by headcount, and hours a week assuming full-time
staff. Time per task is estimated (pipeline/time_shares.py). It is time spent on the tasks, not
time AI would save, and there are no cost estimates.
"""
import json
import re
from collections import Counter
from functools import cache
from pathlib import Path

DATA = Path(__file__).resolve().parent / "data"
BUCKETS = ("E1", "E2", "E0")
HOURS_PER_WEEK = 37.5                  # UK full-time week
TASK_HOURS = HOURS_PER_WEEK * 7 / 8    # of which on the occupation's listed tasks (the rest: breaks, meetings)


@cache
def _load(name):
    return json.loads((DATA / f"{name}.json").read_text())


def meta():
    return _load("industries")["meta"]


def occupations():
    return _load("occupations")["occupations"]


def industries():
    """All industries, without their occupation mix."""
    return [{k: v for k, v in i.items() if k != "mix"} for i in _load("industries")["industries"]]


def industry(code):
    for i in _load("industries")["industries"]:
        if i["code"] == code:
            return i
    raise KeyError(f"unknown industry {code!r}")


def default_mix(code, min_share=0.01, max_rows=30):
    """The industry's typical occupations as percentages summing to 100.

    Occupations under min_share are dropped and the rest scaled up, so the starting list is
    short enough to edit. `covers` says what share of the industry's workers the list holds.
    """
    ind = industry(code)
    keep = [m for m in ind["mix"] if m["share"] >= min_share][:max_rows]
    covers = sum(m["share"] for m in keep)
    occ = occupations()
    rows = [{"soc": m["soc"], "title": occ[m["soc"]]["title"], "value": round(100 * m["share"] / covers, 1)}
            for m in keep]
    return {"industry": {k: v for k, v in ind.items() if k != "mix"}, "covers": round(covers, 3), "rows": rows}


def search(query, limit=20):
    """Find occupations by job title, e.g. 'payroll' or 'people partner'. All words must match."""
    words = [w for w in re.split(r"\W+", query.lower()) if w]
    if not words:
        return []
    occ = occupations()
    hits, examples = Counter(), {}
    for soc, o in occ.items():
        if all(w in o["title"].lower() for w in words):
            hits[soc] += 100                      # the occupation's own name matches
    for title, soc in _load("titles"):
        t = title.lower()
        if all(w in t for w in words):
            hits[soc] += 1
            examples.setdefault(soc, title)
    return [{"soc": soc, "title": occ[soc]["title"], "matched": examples.get(soc)}
            for soc, _ in hits.most_common(limit)]


def scan(rows, headcount=None, mode="percent"):
    """rows: [{"soc": "2412", "value": 20}]. mode "percent": values are shares of headcount
    (scaled to sum to 100); mode "count": values are numbers of people."""
    occ = occupations()
    rows = [r for r in rows if r.get("soc") in occ and float(r.get("value") or 0) > 0]
    total_value = sum(float(r["value"]) for r in rows)
    if mode == "percent":
        if not headcount or headcount <= 0:
            raise ValueError("headcount is required in percent mode")
        people = {r["soc"]: headcount * float(r["value"]) / total_value for r in rows} if total_value else {}
    elif mode == "count":
        people = {r["soc"]: float(r["value"]) for r in rows}
        headcount = sum(people.values())
    else:
        raise ValueError(f"unknown mode {mode!r}")

    out_rows, weighted = [], {b: 0.0 for b in BUCKETS}
    rated_people = unrated_people = 0.0
    for soc, n in people.items():
        o = occ[soc]
        e = o["exposure"]
        row = {"soc": soc, "title": o["title"], "people": round(n, 1), "exposure": e, "beta": o["beta"]}
        if e:
            rated_people += n
            for b in BUCKETS:
                weighted[b] += n * e[b]
        else:
            unrated_people += n
        out_rows.append(row)

    firm = {b: round(weighted[b] / rated_people, 4) if rated_people else None for b in BUCKETS}
    rated = [r for r in out_rows if r["exposure"]]

    def top(bucket, k=5):
        return [{"soc": r["soc"], "title": r["title"], "people": r["people"],
                 "share": r["exposure"][bucket], "hours_each": round(r["exposure"][bucket] * TASK_HOURS, 1),
                 "hours": round(r["people"] * r["exposure"][bucket] * TASK_HOURS)}
                for r in sorted(rated, key=lambda r: -r["people"] * r["exposure"][bucket])[:k]]

    return {
        "headcount": round(headcount or 0, 1),
        "task_hours_per_week": TASK_HOURS,
        "firm": {"shares": firm,
                 "hours": {b: round(weighted[b] * TASK_HOURS) for b in BUCKETS},
                 "beta": round(firm["E1"] + 0.5 * firm["E2"], 4) if rated_people else None,
                 "people_without_data": round(unrated_people, 1)},
        "quick_wins": top("E1"),
        "build_candidates": top("E2"),
        "rows": sorted(out_rows, key=lambda r: -r["people"]),
        "index": meta()["index"],
    }


def tasks(soc):
    """The occupation's tasks with their labels, grouped by bucket (E1 first), longest first.
    hours: estimated hours a week for a full-time worker, or None where there's no time estimate."""
    o = occupations()[soc]
    by_code = _load("tasks")
    out = {b: [] for b in BUCKETS}
    for link in o["onet"]:
        rows = by_code.get(link["code"], [])
        covered = sum(t["share"] or 0 for t in rows)
        for t in rows:
            counts = Counter(t["labels"])
            label = counts.most_common(1)[0][0]
            hours = None
            if link["weighting"] == "time":
                # each linked O*NET occupation counts equally, as in the role's shares
                hours = round((t["share"] or 0) / covered * TASK_HOURS / len(o["onet"]), 1)
            out[label].append({"task": t["task"], "onet": link["title"], "hours": hours,
                               "agreement": round(counts[label] / len(t["labels"]), 2)})
    for b in BUCKETS:
        out[b].sort(key=lambda t: -(t["hours"] or 0))
    return {"soc": soc, "title": o["title"], "onet": o["onet"], "tasks": out}
