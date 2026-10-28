"""Web API, page and MCP server for the scanner.

  uv run uvicorn scanner.app:app --reload     # http://127.0.0.1:8000, MCP at /mcp
"""
import json
import os
import uuid
from functools import cache
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, Field

from langgraph.types import Command

from scanner import ask as ask_agent
from scanner import assess, checkpoints, core, usage
from scanner.mcp_server import server as mcp_server

STATIC = Path(__file__).resolve().parent / "static"
# stateless JSON replies: each MCP request stands alone, as Lambda needs
mcp_app = mcp_server.streamable_http_app(stateless_http=True, json_response=True, host="0.0.0.0")
app = FastAPI(title="Opportunity Scanner", lifespan=lambda _: mcp_server.session_manager.run())
# the page at stablefuture.uk/opportunity calls this API from the browser
ORIGINS = os.environ.get("SCANNER_CORS_ORIGINS",
                         "https://www.stablefuture.uk,https://stablefuture.uk").split(",")
app.add_middleware(CORSMiddleware, allow_origins=ORIGINS, allow_methods=["GET", "POST"],
                   allow_headers=["Content-Type"])


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


@app.post("/api/assess/stream")
def stream_assessment(req: AssessRequest):
    """The same as POST /api/assess, as newline-delimited JSON events while the agent works:
    roles (after parsing), step (each search or read), mapped (each role), then status or error."""
    if not (req.csv or req.job_description):
        raise HTTPException(422, "give a CSV of roles or a job description")
    _take("assess")
    thread_id = str(uuid.uuid4())
    occ = core.occupations()

    def events():
        try:
            for mode, chunk in workflow().stream(req.model_dump(exclude_none=True), _config(thread_id),
                                                  stream_mode=["custom", "updates"]):
                if mode == "custom":
                    yield {"event": "step", **chunk}
                elif "parse" in chunk:
                    yield {"event": "roles", "warnings": chunk["parse"]["warnings"],
                           "roles": [{k: r[k] for k in ("id", "title", "headcount", "department")}
                                     for r in chunk["parse"]["roles"]]}
                elif "map_role" in chunk:
                    for m in chunk["map_role"]["mappings"]:
                        yield {"event": "mapped", **m, "role": occ[m["soc"]]["title"] if m["soc"] else None,
                               "sure": m["soc"] is not None and m["confidence"] >= assess.MIN_CONFIDENCE}
            yield {"event": "status", **_status(thread_id)}
        except ValueError as e:
            yield {"event": "error", "detail": str(e)}

    return StreamingResponse((json.dumps(e) + "\n" for e in events()), media_type="application/x-ndjson",
                             headers={"Cache-Control": "no-store"})


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
