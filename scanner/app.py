"""Web API, page and MCP server for the scanner.

  uv run uvicorn scanner.app:app --reload     # http://127.0.0.1:8000, MCP at /mcp
"""
import uuid
from functools import cache
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from langgraph.types import Command

from scanner import ask as ask_agent
from scanner import assess, checkpoints, core, usage
from scanner.mcp_server import server as mcp_server

STATIC = Path(__file__).resolve().parent / "static"
# stateless JSON replies: each MCP request stands alone, as Lambda needs
mcp_app = mcp_server.streamable_http_app(stateless_http=True, json_response=True, host="0.0.0.0")
app = FastAPI(title="Opportunity Scanner", lifespan=lambda _: mcp_server.session_manager.run())


class Row(BaseModel):
    soc: str = Field(pattern=r"^\d{4}$")
    value: float = Field(ge=0)


class ScanRequest(BaseModel):
    rows: list[Row] = Field(max_length=500)
    headcount: float | None = Field(default=None, gt=0, le=10_000_000)
    mode: Literal["percent", "count"] = "percent"


@app.get("/")
def page():
    return FileResponse(STATIC / "index.html")


@app.get("/api/meta")
def meta():
    return core.meta()


@app.get("/api/industries")
def industries():
    return core.industries()


@app.get("/api/industries/{code}")
def industry_mix(code: str, min_share: float = 0.01):
    try:
        return core.default_mix(code, min_share=min_share)
    except KeyError as e:
        raise HTTPException(404, str(e))


@app.get("/api/search")
def search(q: str):
    return core.search(q[:100])


@app.post("/api/scan")
def scan(req: ScanRequest):
    try:
        return core.scan([r.model_dump() for r in req.rows], req.headcount, req.mode)
    except ValueError as e:
        raise HTTPException(422, str(e))


@app.get("/api/occupations/{soc}/tasks")
def tasks(soc: str):
    if soc not in core.occupations():
        raise HTTPException(404, f"unknown occupation {soc}")
    return core.tasks(soc)


# --- assessment workflow and questions agent (LLM) -----------------------------------------

class AssessRequest(BaseModel):
    csv: str | None = Field(default=None, max_length=200_000)
    job_description: str | None = Field(default=None, max_length=assess.MAX_TEXT)


class ReviewRequest(BaseModel):
    choices: dict[str, str] = Field(max_length=assess.MAX_ROLES)


class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=2000)
    assessment: str | None = Field(default=None, description="thread_id of a finished assessment")


def _take(kind):
    try:
        usage.take(kind)
    except usage.OverCap as e:
        raise HTTPException(429, str(e))


@cache
def workflow():
    return assess.build(checkpoints.saver())


def _config(thread_id):
    return {"configurable": {"thread_id": thread_id}, "max_concurrency": 8}


def _status(thread_id):
    state = workflow().get_state(_config(thread_id))
    if not state.values:
        raise HTTPException(404, "unknown assessment")
    review = next((i.value for t in state.tasks for i in t.interrupts), None)
    if review:
        return {"id": thread_id, "status": "needs_review", "review": review}
    return {"id": thread_id, "status": "done", "report": state.values.get("report")}


@app.post("/api/assess")
def start_assessment(req: AssessRequest):
    if not (req.csv or req.job_description):
        raise HTTPException(422, "give a CSV of roles or a job description")
    _take("assess")
    thread_id = str(uuid.uuid4())
    try:
        workflow().invoke(req.model_dump(exclude_none=True), _config(thread_id))
    except ValueError as e:
        raise HTTPException(422, str(e))
    return _status(thread_id)


@app.post("/api/assess/{thread_id}/review")
def review_assessment(thread_id: str, req: ReviewRequest):
    if _status(thread_id)["status"] != "needs_review":
        raise HTTPException(409, "this assessment isn't waiting for a review")
    workflow().invoke(Command(resume=req.choices), _config(thread_id))
    return _status(thread_id)


@app.get("/api/assess/{thread_id}")
def get_assessment(thread_id: str):
    return _status(thread_id)


@app.post("/api/ask")
def ask(req: AskRequest):
    _take("ask")
    company = None
    if req.assessment:
        done = _status(req.assessment)
        company = done.get("report")
    return ask_agent.ask(req.question, company)


app.mount("/", mcp_app)     # last, so the routes above match first; serves /mcp
