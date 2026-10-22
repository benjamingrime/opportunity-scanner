"""Build the scanner's data files from the raw sources in data/raw.

  uv run python pipeline/build_data.py [--index eloundou-2023-human | path/to/labels/dir]

Writes three JSON files to scanner/data/:
  industries.json   SIC 2007 divisions, each with a typical UK occupation mix (SOC 2020 4-digit)
  occupations.json  SOC 2020 4-digit occupations with exposure shares (E0/E1/E2) and O*NET links
  tasks.json        O*NET tasks with exposure labels, keyed by O*NET code
  titles.json       ONS job titles -> SOC 2020 code, for adding occupations by job title

The index is swappable. Until the 2026 re-rating is complete, the default is Eloundou et al.'s
2023 human labels. A labels directory from gpts-are-gpts-2026 (one CSV per model) gives the
2026 index: each task's label shares are averaged over the models.
"""
import argparse, csv, json, re
from collections import defaultdict
from pathlib import Path

import openpyxl

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
OUT = ROOT / "scanner" / "data"

# Industries whose published cells cover less than this share of their workers borrow the
# occupation mix of their SIC section (the ONS table blanks out small cells).
MIN_COVERAGE = 0.70

SECTIONS = [  # (section, first division, last division, name)
    ("A", 1, 3, "Agriculture, forestry and fishing"),
    ("B", 5, 9, "Mining and quarrying"),
    ("C", 10, 33, "Manufacturing"),
    ("D", 35, 35, "Electricity, gas, steam and air conditioning supply"),
    ("E", 36, 39, "Water supply, sewerage and waste"),
    ("F", 41, 43, "Construction"),
    ("G", 45, 47, "Wholesale and retail trade"),
    ("H", 49, 53, "Transportation and storage"),
    ("I", 55, 56, "Accommodation and food services"),
    ("J", 58, 63, "Information and communication"),
    ("K", 64, 66, "Financial and insurance activities"),
    ("L", 68, 68, "Real estate activities"),
    ("M", 69, 75, "Professional, scientific and technical activities"),
    ("N", 77, 82, "Administrative and support services"),
    ("O", 84, 84, "Public administration and defence"),
    ("P", 85, 85, "Education"),
    ("Q", 86, 88, "Human health and social work"),
    ("R", 90, 93, "Arts, entertainment and recreation"),
    ("S", 94, 96, "Other service activities"),
    ("T", 97, 98, "Households as employers"),
    ("U", 99, 99, "Extraterritorial organisations"),
]


def section_of(division):
    for s, lo, hi, name in SECTIONS:
        if lo <= division <= hi:
            return s, name
    raise ValueError(f"no SIC section for division {division}")


def merge(label):
    return "E2" if label == "E3" else label


def read_industry_table():
    """Pool the 2022-2024 sheets: mean of published values per cell; '-' = 0, '*' = blanked."""
    wb = openpyxl.load_workbook(RAW / "ons-2978" / "4digitsoc20by2digitindustry20222024.xlsx", read_only=True)
    cells = defaultdict(list)          # (division, soc) -> published values
    blank = defaultdict(int)           # (division, soc) -> years blanked
    totals = defaultdict(list)         # division -> industry totals
    divisions, socs = {}, {}
    for ws in wb.worksheets:
        rows = list(ws.iter_rows(values_only=True))
        header = rows[7]
        cols = {}
        for j, h in enumerate(header):
            m = re.match(r"\s*\d+\s+(\d{2})\s+(.*)", str(h or ""))
            if m:
                cols[j] = int(m.group(1))
                divisions[int(m.group(1))] = m.group(2).strip()
        for r in rows[8:]:
            label = str(r[1] or "").strip()
            if label == "Total":
                for j, d in cols.items():
                    if isinstance(r[j], (int, float)):
                        totals[d].append(r[j])
                continue
            m = re.match(r"(\d{4})\s+(.*)", label)
            if not m:
                continue
            soc = m.group(1)
            socs[soc] = m.group(2).strip()
            for j, d in cols.items():
                v = r[j]
                if isinstance(v, (int, float)):
                    cells[(d, soc)].append(v)
                elif v == "-":
                    cells[(d, soc)].append(0)
                elif v == "*":
                    blank[(d, soc)] += 1
    pooled = {k: sum(v) / len(v) for k, v in cells.items() if v}
    total = {d: sum(v) / len(v) for d, v in totals.items() if v}
    return divisions, socs, pooled, total


def build_industries(divisions, socs, pooled, total):
    by_div = defaultdict(dict)
    for (d, soc), v in pooled.items():
        if v > 0:
            by_div[d][soc] = v
    by_sec = defaultdict(lambda: defaultdict(float))
    sec_total = defaultdict(float)
    for d, mix in by_div.items():
        s, _ = section_of(d)
        for soc, v in mix.items():
            by_sec[s][soc] += v
        sec_total[s] += total.get(d, 0)
    out = []
    for d in sorted(divisions):
        if not total.get(d):
            continue
        s, s_name = section_of(d)
        published = sum(by_div[d].values())
        coverage = published / total[d]
        borrowed = coverage < MIN_COVERAGE
        mix = by_sec[s] if borrowed else by_div[d]
        norm = sum(mix.values())
        shares = sorted(((soc, v / norm) for soc, v in mix.items()), key=lambda x: -x[1])
        out.append({
            "code": f"{d:02d}", "name": divisions[d], "section": s, "section_name": s_name,
            "workers": round(total[d]), "coverage": round(coverage, 3),
            "mix_from": f"section {s}" if borrowed else "division",
            "coverage_used": round(sum(by_sec[s].values()) / sec_total[s] if borrowed else coverage, 3),
            "mix": [{"soc": soc, "share": round(sh, 5)} for soc, sh in shares if sh >= 0.0005],
        })
    return out


def load_labels(index):
    """{onet_code: {task_id: (task, [labels from each rater])}} and the index's name."""
    tasks = defaultdict(dict)
    if index == "eloundou-2023-human":
        for r in csv.DictReader(open(RAW / "gpts-are-gpts" / "full_labelset.tsv"), delimiter="\t"):
            tid = str(int(float(r["Task ID"])))
            tasks[r["O*NET-SOC Code"]][tid] = (r["Task"].strip(), [merge(r["human_exposure_agg"])])
        return tasks, "Eloundou et al. (2023), human labels, O*NET 27.2 tasks"
    for path in sorted(Path(index).glob("*.csv")):
        for r in csv.DictReader(open(path)):
            if str(r.get("repeat", "1")) != "1":
                continue
            entry = tasks[r["onet_code"]].setdefault(r["task_id"], (r["task"], []))
            entry[1].append(merge(r["label"]))
    return tasks, f"gpts-are-gpts-2026 ({', '.join(p.stem for p in sorted(Path(index).glob('*.csv')))})"


def shares(labels_per_task):
    """Task-weighted shares of E0/E1/E2. A task rated by several models counts each model equally."""
    n = len(labels_per_task)
    s = {"E0": 0.0, "E1": 0.0, "E2": 0.0}
    for labels in labels_per_task:
        for lab in labels:
            s[lab] += 1 / len(labels) / n
    return s


def build_occupations(socs, tasks):
    crosswalk = json.load(open(RAW / "crosswalk" / "soc4_to_onet.json"))
    titles = {r["O*NET-SOC Code"]: r["Title"] for r in csv.DictReader(
        open(RAW / "onet-31.0" / "Occupation_Data.txt", encoding="utf-8-sig"), delimiter="\t")}
    occ = {}
    for soc, name in sorted(socs.items()):
        onet = [c for c in crosswalk.get(soc, []) if c in tasks]
        per_onet = [shares([labs for _, labs in tasks[c].values()]) for c in onet]
        if per_onet:
            # each linked O*NET occupation counts equally, as Eloundou weight tasks equally
            e = {k: sum(p[k] for p in per_onet) / len(per_onet) for k in ("E0", "E1", "E2")}
            e = {k: round(v, 4) for k, v in e.items()}
            beta = round(e["E1"] + 0.5 * e["E2"], 4)
        else:
            e, beta = None, None
        occ[soc] = {"soc": soc, "title": name, "exposure": e, "beta": beta,
                    "onet": [{"code": c, "title": titles.get(c, c)} for c in onet]}
    return occ


def build_titles(socs):
    wb = openpyxl.load_workbook(RAW / "ons-soc2020" / "soc2020_coding_index.xlsx", read_only=True)
    rows = wb["SOC2020 coding index"].iter_rows(values_only=True)
    head = next(rows)
    i_soc, i_occ, i_add = head.index("SOC_2020"), head.index("INDEXOCC"), head.index("ADD")
    seen, out = set(), []
    for r in rows:
        soc, title = str(r[i_soc] or "").strip(), str(r[i_occ] or "").strip()
        if not (soc in socs and title):
            continue
        if r[i_add]:
            title = f"{title} ({str(r[i_add]).strip()})"
        if (title.lower(), soc) not in seen:
            seen.add((title.lower(), soc))
            out.append([title, soc])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", default="eloundou-2023-human")
    a = ap.parse_args()
    divisions, socs, pooled, total = read_industry_table()
    industries = build_industries(divisions, socs, pooled, total)
    tasks, index_name = load_labels(a.index)
    occupations = build_occupations(socs, tasks)
    task_out = {c: [{"id": tid, "task": t, "labels": labs} for tid, (t, labs) in sorted(v.items())]
                for c, v in tasks.items()}
    OUT.mkdir(parents=True, exist_ok=True)
    meta = {"index": index_name, "industry_source": "ONS ad hoc 2978, APS 2022-2024 pooled",
            "min_coverage": MIN_COVERAGE}
    for name, obj in [("industries", {"meta": meta, "industries": industries}),
                      ("occupations", {"meta": meta, "occupations": occupations}),
                      ("tasks", task_out), ("titles", build_titles(socs))]:
        (OUT / f"{name}.json").write_text(json.dumps(obj, separators=(",", ":")))
    borrowed = sum(i["mix_from"] != "division" for i in industries)
    missing = sum(o["exposure"] is None for o in occupations.values())
    print(f"{len(industries)} industries ({borrowed} borrow their section's mix), "
          f"{len(occupations)} occupations ({missing} without exposure data), index: {index_name}")


if __name__ == "__main__":
    main()
