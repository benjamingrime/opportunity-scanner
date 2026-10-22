"""Read-only tools shared by the assessment workflow, the questions agent and the MCP server.

Plain functions, no LLM inside. Roles are UK SOC 2020 unit groups (four-digit codes); job titles
come from the ONS SOC 2020 coding index.
"""
import re
from collections import Counter

from scanner import core

MAX_MATCHES = 50


def list_roles():
    """All roles as (code, title), in code order."""
    return sorted((soc, o["title"]) for soc, o in core.occupations().items())


def grep_titles(pattern, limit=MAX_MATCHES, hidden=frozenset()):
    """Search the ~30,000 ONS job titles and the 412 role names, case-insensitive.

    pattern is a regular expression ("people partner", "^payroll", "solicitor|lawyer"); if it
    isn't valid, it's searched as plain text. hidden: lowercased titles to leave out (evals hold
    some back so the model can't look up its own answers).
    """
    try:
        rx = re.compile(pattern, re.I)
    except re.error:
        rx = re.compile(re.escape(pattern), re.I)
    occ = core.occupations()
    hits, seen = [], set()
    for soc, title in [(s, o["title"]) for s, o in occ.items()] + [(s, t) for t, s in core._load("titles")]:
        key = (title.lower(), soc)
        if key in seen or title.lower() in hidden or not rx.search(title):
            continue
        seen.add(key)
        hits.append({"title": title, "soc": soc, "role": occ[soc]["title"]})
    hits.sort(key=lambda h: len(h["title"]))
    by_role = Counter(h["soc"] for h in hits)
    return {"total": len(hits),
            "matches": hits[:limit],
            "by_role": [{"soc": s, "role": occ[s]["title"], "titles": n} for s, n in by_role.most_common(10)]}


def get_role(soc, tasks_per_heading=8):
    """A role's exposure split, its O*NET links and its longest tasks under each heading."""
    occ = core.occupations()
    if soc not in occ:
        raise KeyError(f"unknown role {soc!r}")
    o = occ[soc]
    t = core.tasks(soc)["tasks"]
    return {"soc": soc, "title": o["title"], "exposure": o["exposure"], "beta": o["beta"],
            "onet": [{"code": l["code"], "title": l["title"]} for l in o["onet"]],
            "tasks": {b: [{"task": x["task"], "hours_a_week": x["hours"]} for x in t[b][:tasks_per_heading]]
                      for b in core.BUCKETS},
            "index": core.meta()["index"]}


def compare_roles(socs):
    """Exposure side by side for up to 20 roles."""
    occ = core.occupations()
    return [{"soc": s, "title": occ[s]["title"], "exposure": occ[s]["exposure"], "beta": occ[s]["beta"]}
            for s in socs[:20] if s in occ]


def scan(roles, headcount=None, mode="count"):
    """roles: [{"soc": "2412", "value": 3}]. mode "count": values are people; "percent": shares
    of headcount, which is then required."""
    return core.scan(roles, headcount, mode)
