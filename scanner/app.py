"""Web API and page for the scanner.

  uv run uvicorn scanner.app:app --reload     # http://127.0.0.1:8000
"""
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from scanner import core

STATIC = Path(__file__).resolve().parent / "static"
app = FastAPI(title="Opportunity Scanner")


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
