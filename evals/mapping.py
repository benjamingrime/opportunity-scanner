"""Mapping eval: does the workflow find the right UK job title?

Test titles come from the ONS SOC 2020 coding index: one title from each of n randomly chosen unit
groups, among titles coded to exactly one unit group, rewritten the way people write them
("Accounts clerk", not "Clerk, accounts"). They are hidden from grep_titles during the run, so the
model can't look up its own answers.

  uv run python -m evals.mapping --model bedrock:eu.amazon.nova-2-lite-v1:0 --n 100
  uv run python -m evals.mapping --replay            # recorded replies, no model calls
"""
import argparse
import json
import random
import re
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from scanner import assess, core, llm
from scanner.replay import Replay

ROOT = Path(__file__).resolve().parent
PRICES = {  # USD per million tokens: input, output, cached input. OpenRouter and Bedrock eu-north-1, 22 Oct 2026
    "openrouter:anthropic/claude-haiku-5.5": (0.10, 0.50, 0.01),
    "bedrock:eu.amazon.nova-2-lite-v1:0": (0.363, 2.992, 0.0908),
    "bedrock:openai.gpt-oss-120b-1:0": (0.15, 0.60, 0.15),
    "bedrock:qwen.qwen3-235b-a22b-2507-v1:0": (0.22, 0.88, 0.22),
    "bedrock:deepseek.v3.2": (0.74, 2.22, 0.74),
}


def slug(spec):
    return re.sub(r"[^a-z0-9.]+", "-", spec.lower()).strip("-")


def natural(title):
    """'Clerk, accounts' -> 'Accounts clerk'."""
    parts = [p.strip() for p in title.split(",") if p.strip()]
    t = " ".join(reversed(parts))
    return t[0].upper() + t[1:].lower() if not re.search(r"\b[A-Z]{2,}\b", t) else t[0].upper() + t[1:]


def sample(n=100, seed=1):
    socs_by_title = defaultdict(set)
    originals = defaultdict(list)
    for title, soc in core._load("titles"):
        if "(" in title:
            continue                                   # qualified titles lean on their qualifier
        socs_by_title[title.lower()].add(soc)
        originals[soc].append(title)
    rng = random.Random(seed)
    cases = []
    for soc in rng.sample(sorted(originals), n):
        pool = sorted({t for t in originals[soc] if socs_by_title[t.lower()] == {soc}})
        if pool:
            t = rng.choice(pool)
            cases.append({"id": f"e{len(cases) + 1}", "title": natural(t), "index_title": t, "soc": soc,
                          "headcount": 1.0, "department": None, "description": None})
    return cases


def run(spec, n=100, seed=1, model=None, workers=8):
    cases = sample(n, seed)
    hidden = frozenset(c["index_title"].lower() for c in cases)
    t0 = time.time()
    with ThreadPoolExecutor(workers) as pool:
        maps = list(pool.map(lambda c: assess.map_one(c, model=model, hidden=hidden, spec=spec), cases))
    secs = time.time() - t0
    rows = []
    for c, m in zip(cases, maps):
        rows.append({"title": c["title"], "want": c["soc"], "got": m["soc"], "confidence": m["confidence"],
                     "alternatives": m["alternatives"], "searches": m["searches"], "reason": m["reason"]})
    return rows, secs


def summarise(spec, rows, secs, usage):
    n = len(rows)
    right = [r for r in rows if r["got"] == r["want"]]
    sure = [r for r in rows if r["got"] and r["confidence"] >= assess.MIN_CONFIDENCE]
    p_in, p_out, p_cache = PRICES.get(spec, (0, 0, 0))
    cached = usage.get("cache_read", 0)
    cost = ((usage.get("input", 0) - cached) * p_in + cached * p_cache + usage.get("output", 0) * p_out) / 1e6
    return {
        "model": spec, "n": n,
        "exact": round(len(right) / n, 3),
        "top3": round(sum(r["want"] in [r["got"], *r["alternatives"]] for r in rows) / n, 3),
        "same_minor_group": round(sum((r["got"] or "")[:3] == r["want"][:3] for r in rows) / n, 3),
        "sure_share": round(len(sure) / n, 3),
        "exact_when_sure": round(sum(r["got"] == r["want"] for r in sure) / len(sure), 3) if sure else None,
        "searches_per_role": round(sum(len(r["searches"]) for r in rows) / n, 2),
        "tokens": usage, "cost_usd": round(cost, 4), "cost_per_role_usd": round(cost / n, 5),
        "seconds": round(secs, 1),
    }


class Counting:
    """Wraps a chat model and adds up its token use."""
    def __init__(self, model, usage=None, tools=None):
        self.model, self.usage, self.tools = model, usage if usage is not None else defaultdict(int), tools

    def bind_tools(self, tools):
        return Counting(self.model, self.usage, tools)

    def invoke(self, messages):
        m = self.model.bind_tools(self.tools) if self.tools else self.model
        ai = m.invoke(messages)
        u = ai.usage_metadata or {}
        self.usage["input"] += u.get("input_tokens", 0)
        self.usage["output"] += u.get("output_tokens", 0)
        self.usage["cache_read"] += (u.get("input_token_details") or {}).get("cache_read", 0)
        return ai


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=llm.name())
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--record", action="store_true", help="record replies for replay in tests and CI")
    ap.add_argument("--replay", action="store_true", help="replay recorded replies; no model calls")
    a = ap.parse_args()
    path = ROOT / "recordings" / f"{slug(a.model)}.json.gz"
    if a.replay:
        inner = Replay(path)
    elif a.record:
        inner = Replay(path, llm.chat_model(a.model))
    else:
        inner = llm.chat_model(a.model)
    model = Counting(inner)
    rows, secs = run(a.model, a.n, a.seed, model, a.workers)
    if a.record:
        inner.save()
    s = summarise(a.model, rows, secs, dict(model.usage))
    out = ROOT / "results" / f"mapping-{slug(a.model)}.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps({"summary": s, "rows": rows}, indent=1))
    print(json.dumps(s, indent=1))


if __name__ == "__main__":
    main()
