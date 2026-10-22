"""Faithfulness check: every number in an answer must come from tool output or the question.

Numbers may be rounded or written as percentages (0.2312 -> 23%, 23.1%). Whole numbers up to 10
without a % sign are left alone: they're counts in the prose ("the top 3 roles").
"""
import json
import re

NUMBER = re.compile(r"(?<![\w.])-?\d[\d,]*(?:\.\d+)?(?:\s?%)?")


def numbers_in(text):
    out = []
    for m in NUMBER.finditer(text):
        raw = m.group().replace(",", "").replace(" ", "")
        pct = raw.endswith("%")
        num = raw.rstrip("%")
        dp = len(num.split(".")[1]) if "." in num else 0
        out.append((m.group().strip(), float(num), dp, pct))
    return out


def _walk(x):
    if isinstance(x, bool):
        return
    if isinstance(x, (int, float)):
        yield float(x)
    elif isinstance(x, str):
        for _, v, _, _ in numbers_in(x):
            yield v
    elif isinstance(x, dict):
        for v in x.values():
            yield from _walk(v)
    elif isinstance(x, (list, tuple)):
        for v in x:
            yield from _walk(v)


def sources(*texts_or_objects):
    vals = set()
    for t in texts_or_objects:
        if isinstance(t, str):
            try:
                t = json.loads(t)
            except ValueError:
                pass
        vals.update(_walk(t))
    return vals


def unsupported(answer, source_values):
    """Numbers in the answer that no source value rounds to."""
    bad = []
    for raw, n, dp, pct in numbers_in(answer):
        if not pct and dp == 0 and abs(n) <= 10:
            continue
        tol = 0.5 * 10 ** -dp + 1e-9
        cands = [v * 100 for v in source_values] + list(source_values) if pct else list(source_values)
        if not any(abs(n - v) <= tol for v in cands):
            bad.append(raw)
    return bad
