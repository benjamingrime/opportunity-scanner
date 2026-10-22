"""Estimate how working time splits across the tasks of each O*NET occupation.

Method: Hatgis-Kessell, Aguirre, Wan and Bommasani (2026), "Estimating time spent on work tasks",
arXiv:2608.05172, https://github.com/Stephanehk/Estimating-Time-Spent-On-Work. Reimplemented here
from the paper.

  hours a day on a task = (times done a day, from O*NET's frequency ratings) x (hours one go takes)

Hours per go are never asked for directly. Within each occupation, a model judges which of two
tasks takes longer to do once. Copeland's method turns the judgements into a ranking, and a linear
program finds hours per go that respect the ranking and fill a 7-hour day on tasks.

Where we differ from the paper:
- Model: Claude Haiku 5.5 at high reasoning, via OpenRouter (paper: GPT-5.2, no reasoning).
- One call per task: the model compares that task with every other task in its occupation, so
  each pair is judged twice, once from each side, in separate calls. A pair counts only when the
  two judgements agree. This replaces the paper's filter on token probability (>= 0.7), which
  Anthropic models don't expose.
- Fixed frequency constants (260 working days a year), not per-occupation US CPS hours.
- O*NET 31.0 (paper: 30.2).

  uv run python pipeline/time_shares.py compare --only 23-1011.00 15-1252.00   # pilot
  uv run python pipeline/time_shares.py compare --cap 25                         # all occupations
  uv run python pipeline/time_shares.py export   # cache -> data/time/judgments.csv.gz, run.json
  uv run python pipeline/time_shares.py solve    # judgments + ratings -> data/time/time_shares.csv

Every reply is cached under a hash of (model, effort, prompt, occupation, task, the other tasks),
in a folder per O*NET major version. In a monthly release (31.1, 31.2, ...) only occupations whose
task lists changed are compared afresh; the rest are cache hits. A new .0 release starts a fresh
folder, so every occupation is re-estimated. `solve` always runs on everything: it is free.
`compare` needs OPENROUTER_API_KEY.
"""
import argparse, csv, gzip, hashlib, http.client, io, json, os, re, time, urllib.error, urllib.request
from collections import defaultdict
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path

import numpy as np
from scipy.optimize import linprog

ROOT = Path(__file__).resolve().parents[1]
ONET_RELEASE = "31.0"
ONET = ROOT / "data" / "raw" / f"onet-{ONET_RELEASE}"
CACHE = ROOT / "data" / "cache" / "time"
OUT = ROOT / "data" / "time"
MODEL = "anthropic/claude-haiku-5.5"
EFFORT = "high"

# O*NET frequency category -> times done a day. 0 = never (imputed from the relevance rating).
PER_DAY = {0: 0, 1: 1 / 520, 2: 1 / 260, 3: 1 / 20, 4: 1 / 5, 5: 1, 6: 2, 7: 7}
HOURS_PER_DAY = 7  # hours a day on O*NET tasks, of an 8-hour day (the rest: breaks, meetings)
MARGIN = 0.1       # a task ranked longer takes at least this many more hours per go
MIN_HOURS = 0.1    # the shortest go of any task, in hours

SYSTEM = """You compare how long work tasks take within one occupation.

You get an occupation, one task (task A) and a numbered list of other tasks from the same occupation. For each listed task, judge which takes a typical worker in this occupation longer under normal working conditions: one go of task A, or one go of the listed task (B).

Rules:
1. Ignore how often a task is done. Judge the length of one go only.
2. One go is one run-through of the task as written. Cleaning a home: cleaning it once. Answering calls: answering one call. Monitoring staff for safety breaches: one round of monitoring. For a task that runs over many sessions, such as writing a book, one go is the time a typical worker spends making real progress on it in a day.
3. Leave out waiting or idle time, unless the worker must stay actively engaged.
4. Assume the worker has no generative AI help.
5. Answer "same" if they take roughly the same time. Answer "unsure" if you can't tell, would need more context, or the answer depends on the situation.

Return only JSON, one entry for every listed task, in order:
{"answers": [{"n": 1, "longer": "A" | "B" | "same" | "unsure"}, ...]}"""
ANSWERS = {"a", "b", "same", "unsure"}


def read(path):
    return csv.DictReader(open(path, encoding="utf-8-sig"), delimiter="\t")


def load_tasks():
    """{code: (title, [(task_id, task, times a day)])}, for tasks with frequency and relevance ratings."""
    titles = {r["O*NET-SOC Code"]: r["Title"] for r in read(ONET / "Occupation_Data.txt")}
    text = {(r["O*NET-SOC Code"], r["Task ID"]): r["Task"].strip() for r in read(ONET / "Task_Statements.txt")}
    freq, rel = defaultdict(dict), {}
    for r in read(ONET / "Task_Ratings.txt"):
        k = (r["O*NET-SOC Code"], r["Task ID"])
        if r["Scale ID"] == "FT":
            freq[k][int(r["Category"])] = float(r["Data Value"])
        elif r["Scale ID"] == "RT":
            rel[k] = float(r["Data Value"])
    occ = defaultdict(list)
    for k, dist in freq.items():
        if k not in rel or k not in text:
            continue
        # Frequency answers come only from workers who said the task is relevant; the rest never do it.
        dist = {**dist, 0: 100 - rel[k]}
        total = sum(dist.values())
        occ[k[0]].append((k[1], text[k], sum(PER_DAY[c] * v / total for c, v in dist.items())))
    return {c: (titles[c], sorted(ts, key=lambda t: int(t[0]))) for c, ts in sorted(occ.items()) if len(ts) >= 2}


# --- compare: one call per task -----------------------------------------------------------------

def jobs_for(code, title, tasks):
    for i, anchor in enumerate(tasks):
        yield code, title, anchor, tasks[:i] + tasks[i + 1:]


def cache_path(code, title, anchor, others):
    key = json.dumps([MODEL, EFFORT, SYSTEM, code, title, anchor[:2], [o[:2] for o in others]])
    folder = CACHE / MODEL.replace("/", "_") / EFFORT / f"onet-{ONET_RELEASE.split('.')[0]}" / code
    return folder / f"{hashlib.sha256(key.encode()).hexdigest()[:24]}.json"


def parse(content, n):
    m = re.search(r"\{.*\}", content, re.S)
    if not m:
        raise ValueError("no JSON in reply")
    got = {int(a["n"]): str(a["longer"]).strip().lower() for a in json.loads(m.group(0))["answers"]}
    if sorted(got) != list(range(1, n + 1)) or not set(got.values()) <= ANSWERS:
        raise ValueError(f"expected {n} answers in {sorted(ANSWERS)}, got {len(got)}")
    return [got[i] for i in range(1, n + 1)]


def call(title, anchor, others):
    listed = "\n".join(f"{i}. {o[1]}" for i, o in enumerate(others, 1))
    body = {"model": MODEL, "reasoning": {"effort": EFFORT}, "usage": {"include": True},
            "messages": [{"role": "system", "content": SYSTEM},
                         {"role": "user", "content": f"Occupation: {title}\n\nTask A: {anchor[1]}\n\nListed tasks:\n{listed}"}]}
    req = urllib.request.Request(
        "https://openrouter.ai/api/v1/chat/completions", json.dumps(body).encode(),
        {"Authorization": f"Bearer {os.environ['OPENROUTER_API_KEY']}", "Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as r:
        return json.load(r)


def compare_one(job, tries=6):
    code, title, anchor, others = job
    path = cache_path(*job)
    if path.exists():
        return json.loads(path.read_text()), False
    err = None
    for attempt in range(tries):
        try:
            d = call(title, anchor, others)
            u, choice = d.get("usage") or {}, d["choices"][0]
            # The host's content filter sometimes refuses harmless tasks (e.g. pesticide spraying).
            # A refusal counts as "unsure" on every pair, so those pairs go undecided.
            refused = choice.get("finish_reason") == "content_filter"
            rec = {"onet_code": code, "task_id": anchor[0], "others": [o[0] for o in others],
                   "answers": ["unsure"] * len(others) if refused else parse(choice["message"]["content"], len(others)),
                   "refused": refused,
                   "model": d.get("model"), "host": d.get("provider"), "effort": EFFORT,
                   "prompt_tokens": u.get("prompt_tokens"), "completion_tokens": u.get("completion_tokens"),
                   "reasoning_tokens": (u.get("completion_tokens_details") or {}).get("reasoning_tokens"),
                   "cost": u.get("cost") or 0.0, "attempts": attempt + 1,
                   "compared_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(rec))
            return rec, True
        except (OSError, http.client.HTTPException, ValueError, KeyError, IndexError, TypeError) as e:
            err = e
            pause = 2 ** attempt
            if isinstance(e, urllib.error.HTTPError) and e.code == 429:
                pause = float(e.headers.get("Retry-After") or min(60, 10 * 2 ** attempt))
            time.sleep(pause)
    return {"onet_code": code, "task_id": anchor[0], "error": f"{type(err).__name__}: {err}"[:300]}, False


def compare(occs, cap, workers):
    """Keep `workers` calls in flight; stop starting new ones once settled spend reaches the cap."""
    jobs = iter([j for code, (title, tasks) in occs.items() for j in jobs_for(code, title, tasks)])
    spent, done, new, failed, pending = 0.0, 0, 0, [], set()
    print(f"{len(occs)} occupations, {sum(len(t) for _, t in occs.values())} calls (cached ones are free)", flush=True)
    with ThreadPoolExecutor(workers) as ex:
        while True:
            while len(pending) < workers and spent < cap and (job := next(jobs, None)):
                pending.add(ex.submit(compare_one, job))
            if not pending:
                break
            finished, pending = wait(pending, return_when=FIRST_COMPLETED)
            for f in finished:
                rec, fresh = f.result()
                if "error" in rec:
                    failed.append(rec)
                    continue
                done += 1
                new += fresh
                spent += rec["cost"] if fresh else 0
                if done % 1000 == 0:
                    print(f"  {done} done  new {new}  ${spent:.3f}", flush=True)
    if spent >= cap:
        print(f"  stopped at spend cap ${cap:.2f}")
    print(f"{done} done ({new} new), {len(failed)} failed, new spend ${spent:.4f}")
    for e in failed[:5]:
        print("  failed:", e)


# --- export: cache -> committed judgements --------------------------------------------------------

def export(occs):
    """Write judgements for occupations whose calls are all cached. Gzip without a timestamp, so
    the same judgements give the same bytes."""
    rows, stats = [], defaultdict(float)
    for code, (title, tasks) in occs.items():
        recs = [cache_path(*j) for j in jobs_for(code, title, tasks)]
        if not all(p.exists() for p in recs):
            stats["incomplete_occupations"] += 1
            continue
        stats["occupations"] += 1
        for p in recs:
            r = json.loads(p.read_text())
            stats["calls"] += 1
            stats["refused"] += r.get("refused", False)
            for k in ("cost", "prompt_tokens", "completion_tokens", "reasoning_tokens"):
                stats[k] += r.get(k) or 0
            rows += [(code, r["task_id"], other, ans) for other, ans in zip(r["others"], r["answers"])]
    OUT.mkdir(parents=True, exist_ok=True)
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator="\n")
    w.writerow(["onet_code", "task_id", "other_id", "longer"])
    w.writerows(rows)
    with open(OUT / "judgments.csv.gz", "wb") as f, gzip.GzipFile(fileobj=f, mode="wb", mtime=0) as g:
        g.write(buf.getvalue().encode())
    run = {"model": MODEL, "effort": EFFORT, "onet": ONET_RELEASE, **{k: round(v, 4) if k == "cost" else int(v)
                                                               for k, v in stats.items()}}
    (OUT / "run.json").write_text(json.dumps(run, indent=2) + "\n")
    print(json.dumps(run))


# --- solve: judgements -> hours ------------------------------------------------------------------

def decided_pairs(ids, judg):
    """{(i, j): index of the longer task} for pairs where both calls agree. judg[(a, b)] is the
    answer from the call anchored on a: 'a' = a takes longer, 'b' = b takes longer."""
    out, seen = {}, 0
    for i in range(len(ids)):
        for j in range(i + 1, len(ids)):
            x, y = judg.get((ids[i], ids[j])), judg.get((ids[j], ids[i]))
            seen += 1
            if x == "a" and y == "b":
                out[(i, j)] = i
            elif x == "b" and y == "a":
                out[(i, j)] = j
    return out, seen


def solve_occupation(per_day, pairs, objective="span", scale=1.0):
    """Hours per go for each task, or None. Follows the paper: Copeland scores (wins) order the
    agreed pairs; if infeasible, ranking constraints are dropped from the bottom of the ranking up.
    Objective "span" (paper's main one) maximises the gap between the top- and bottom-ranked
    task's hours per go; "even" maximises the least time a day spent on any task. The paper shows
    the two as the edges of a range; "mid" is their average, which also meets every constraint.
    scale shrinks the minimum hours per go and the ranking margin together."""
    if objective == "mid":
        (a, k), (b, _) = (solve_occupation(per_day, pairs, o, scale) for o in ("span", "even"))
        return (None, None) if a is None else ((a + b) / 2, k)
    n = len(per_day)
    score = [0] * n
    for winner in pairs.values():
        score[winner] += 1
    rows = {}
    for (i, j) in pairs:
        if score[i] != score[j]:
            hi, lo = (i, j) if score[i] > score[j] else (j, i)
            row = np.zeros(n + 1)  # hours per go for each task, then t (least hours a day on a task)
            row[lo], row[hi] = 1, -1  # w_lo - w_hi <= -MARGIN
            rows[(i, j)] = row
    top = max(range(n), key=lambda i: (score[i], -i))
    bottom = min(range(n), key=lambda i: (score[i], i))
    c = np.zeros(n + 1)
    floor = []  # t <= w_i * per_day_i
    if objective == "span":
        c[top], c[bottom] = -1, 1
    else:
        c[n] = -1
        for i in range(n):
            row = np.zeros(n + 1)
            row[i], row[n] = -per_day[i], 1
            floor.append(row)
    kw = dict(A_eq=[np.append(per_day, 0)], b_eq=[HOURS_PER_DAY],
              bounds=[(MIN_HOURS * scale, HOURS_PER_DAY)] * n + [(0, None)], method="highs")

    def attempt(keep):
        A = floor + [rows[k] for k in keep]
        b = [0] * len(floor) + [-MARGIN * scale] * len(keep)
        return linprog(c, A_ub=A or None, b_ub=b or None, **kw)

    res = attempt(sorted(rows))
    if res.success:
        return res.x[:n], 0
    order = sorted(range(n), key=lambda i: (score[i], i))
    for k in range(1, n):
        removed, boundary = set(order[:k]), tuple(sorted((order[k - 1], order[k])))
        res = attempt([p for p in sorted(rows) if not (set(p) & removed) or p == boundary])
        if res.success:
            return res.x[:n], k
    return None, None


def solve(occs, objective):
    judg = defaultdict(dict)
    with gzip.open(OUT / "judgments.csv.gz", "rt") as f:
        for r in csv.DictReader(f):
            judg[r["onet_code"]][(r["task_id"], r["other_id"])] = r["longer"]
    rows, unsolved, relaxed, rescaled, n_pairs, n_agreed = [], [], 0, [], 0, 0
    for code, (title, tasks) in occs.items():
        if code not in judg:
            continue
        ids = [t[0] for t in tasks]
        per_day = np.array([t[2] for t in tasks])
        pairs, seen = decided_pairs(ids, judg[code])
        n_pairs, n_agreed = n_pairs + seen, n_agreed + len(pairs)
        w, k = solve_occupation(per_day, pairs, objective)
        # Some jobs' tasks are so frequent (waiters, air traffic controllers) that even the
        # minimum time per go overflows the day; shrink the minimum and margin until it fits.
        for scale in (0.5, 0.25, 0.1):
            if w is not None:
                break
            w, k = solve_occupation(per_day, pairs, objective, scale)
            rescaled += [code] if w is not None else []
        if w is None:
            unsolved.append(code)
            continue
        relaxed += k > 0
        for (tid, _, f), hours in zip(tasks, w):
            rows.append([code, tid, f"{f:.6f}", f"{hours:.4f}", f"{hours * f:.4f}", f"{hours * f / HOURS_PER_DAY:.5f}"])
    with open(OUT / "time_shares.csv", "w", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["onet_code", "task_id", "times_per_day", "hours_per_go", "hours_per_day", "share"])
        w.writerows(rows)
    summary = {"objective": objective, "occupations": len({r[0] for r in rows}), "tasks": len(rows), "relaxed": relaxed,
               "rescaled": rescaled, "unsolved": unsolved, "pairs": n_pairs, "agreed_pairs": n_agreed,
               "agreement": round(n_agreed / n_pairs, 4) if n_pairs else None}
    (OUT / "solve.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["compare", "export", "solve"])
    ap.add_argument("--only", nargs="*", help="O*NET codes (default: all)")
    ap.add_argument("--cap", type=float, default=5.0, help="stop new calls once this much is spent (USD)")
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--effort", default=EFFORT)
    ap.add_argument("--objective", choices=["span", "even", "mid"], default="mid")
    a = ap.parse_args()
    globals()["EFFORT"] = a.effort
    occs = load_tasks()
    if a.only:
        occs = {c: occs[c] for c in a.only}
    {"compare": lambda: compare(occs, a.cap, a.workers), "export": lambda: export(occs),
     "solve": lambda: solve(occs, a.objective)}[a.step]()


if __name__ == "__main__":
    main()
