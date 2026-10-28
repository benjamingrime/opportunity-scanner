"""The assessment workflow and the tools. LLM replies are replayed from recordings, so these run
offline and free. To record again (after changing a prompt, a tool or the index):

  RECORD=1 uv run pytest tests/test_assess.py        # needs OPENROUTER_API_KEY
  uv run python -m evals.mapping --record
"""
import os
import uuid
from pathlib import Path

import pytest
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from evals import mapping
from scanner import assess, llm, tools
from scanner.replay import Replay

HERE = Path(__file__).resolve().parent
HAIKU = "openrouter:anthropic/claude-haiku-5.5"


@pytest.fixture
def model():
    path = HERE / "recordings" / "assess-haiku.json.gz"
    m = Replay(path, llm.chat_model(HAIKU)) if os.environ.get("RECORD") else Replay(path)
    yield m
    if os.environ.get("RECORD"):
        m.save()


def test_grep_titles_finds_inverted_index_titles():
    r = tools.grep_titles("personnel")
    assert r["total"] > 5
    assert {"soc": "3571", "role": "Human resources and industrial relations officers", "titles": 7} in r["by_role"]
    assert tools.grep_titles("(unclosed")["total"] == 0          # bad regex is searched as text


def test_grep_titles_hides_held_out_titles():
    assert any(m["title"] == "Clerk, personnel" for m in tools.grep_titles("clerk, personnel")["matches"])
    hidden = tools.grep_titles("clerk, personnel", hidden=frozenset({"clerk, personnel"}))
    assert all(m["title"] != "Clerk, personnel" for m in hidden["matches"])


def test_csv_drops_personal_data_and_reads_headcount():
    roles, warnings = assess.parse_csv("Name,Job title,Headcount,Email\nAna,Paralegal,3,a@x.com\nBo,Clerk,x,b@x.com\n")
    assert [r["title"] for r in roles] == ["Paralegal", "Clerk"]
    assert [r["headcount"] for r in roles] == [3.0, 1.0]
    assert "Name" in warnings[0] and "Email" in warnings[0]
    assert all("Ana" not in str(r) and "@" not in str(r) for r in roles)


def test_csv_needs_a_title_column():
    with pytest.raises(ValueError, match="title"):
        assess.parse_csv("Headcount\n3\n")


def test_workflow_pauses_for_unsure_roles_then_scores(model):
    graph = assess.build(InMemorySaver())
    config = {"configurable": {"thread_id": str(uuid.uuid4()), "model": model}}
    out = graph.invoke({"csv": (HERE / "fixtures" / "roles.csv").read_text()}, config)

    ask = out["__interrupt__"][0].value
    unsure = {r["title"] for r in ask["roles"]}
    assert {"Senior Associate", "Analyst"} <= unsure                 # vague titles go to a person
    assert "Payroll Administrator" not in unsure
    for r in ask["roles"]:
        assert r["candidates"]

    choices = {r["id"]: r["candidates"][-1]["soc"] for r in ask["roles"]}
    report = graph.invoke(Command(resume=choices), config)["report"]
    rows = {r["title"]: r for r in report["roles"]}
    assert rows["Payroll Administrator"]["soc"] == "4122"
    for r in ask["roles"]:
        row = next(x for x in report["roles"] if x["id"] == r["id"])
        assert (row["soc"], row["chosen_by"]) == (choices[r["id"]], "person")
    assert report["scan"]["headcount"] == 35
    assert sum(report["scan"]["firm"]["shares"].values()) == pytest.approx(1, abs=1e-3)
    assert "Employee email" in report["warnings"][0]


def test_mapping_eval_replayed():
    """The recorded 100-title eval: a regression check on prompt and tool changes."""
    spec = HAIKU
    rows, _ = mapping.run(spec, n=100, model=Replay(mapping.ROOT / "recordings" / f"{mapping.slug(spec)}.json.gz"))
    s = mapping.summarise(spec, rows, 0, {})
    assert s["exact"] >= 0.65
    assert s["top3"] >= 0.8
    assert s["exact_when_sure"] >= 0.85          # a sure mapping skips review, so it must be right


def test_api_assessment_round_trip(client, model, monkeypatch):
    monkeypatch.setattr(llm, "chat_model", lambda *a, **k: model)
    monkeypatch.delenv("SCANNER_LLM", raising=False)
    c = client
    r = c.post("/api/assess", json={"csv": (HERE / "fixtures" / "roles.csv").read_text()}).json()
    assert r["status"] == "needs_review"
    choices = {x["id"]: x["candidates"][0]["soc"] for x in r["review"]["roles"]}
    done = c.post(f"/api/assess/{r['id']}/review", json={"choices": choices}).json()
    assert done["status"] == "done" and len(done["report"]["roles"]) == 8
    assert c.get(f"/api/assess/{r['id']}").json()["status"] == "done"
    assert c.post(f"/api/assess/{r['id']}/review", json={"choices": {}}).status_code == 409
    answer = c.post("/api/ask", json={"question": "Which of our roles has the largest share of buy-and-train work?",
                                      "assessment": r["id"]}).json()
    assert answer["answer"] and answer["unsupported"] == []


def test_api_assessment_streams_the_agent_at_work(client, model, monkeypatch):
    import json
    monkeypatch.setattr(llm, "chat_model", lambda *a, **k: model)
    r = client.post("/api/assess/stream", json={"csv": (HERE / "fixtures" / "roles.csv").read_text()},
                    headers={"Origin": "https://www.stablefuture.uk"})
    assert r.headers["access-control-allow-origin"] == "https://www.stablefuture.uk"
    events = [json.loads(line) for line in r.text.splitlines()]
    kinds = [e["event"] for e in events]
    assert kinds[0] == "roles" and len(events[0]["roles"]) == 8
    assert kinds.count("mapped") == 8 and "step" in kinds
    assert all(e["pattern"] for e in events if e["event"] == "step" and e["kind"] == "search")
    assert kinds[-1] == "status" and events[-1]["status"] == "needs_review"
