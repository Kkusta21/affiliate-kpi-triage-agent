"""Small HTTP API and a one page UI on top of the same workflow.

    uvicorn triage_agent.api:app --port 8000
    POST /triage      body: alert JSON   -> TriageResult (200 completed, 422 rejected, 404 unknown campaign, 500 failed)
    GET  /incidents                      -> incident list
    GET  /samples                        -> sample alerts for the UI
    GET  /health                         -> liveness and dependency check, no auth
    GET  /                               -> demo page
    GET  /docs                           -> OpenAPI documentation

The API layer only translates between HTTP and the agent. Validation, refusals and the
verdict all live in the workflow, so the CLI, the API and the tests share one code path.
The response body is always a TriageResult, whatever the status code.

Endpoints are plain `def` on purpose: a triage run is blocking work (tools, SQLite, model
calls), so FastAPI runs it in its thread pool and the event loop stays free.

Authorization happens here, outside the model: set TRIAGE_API_KEY to require the header
`X-API-Key` on every data endpoint. When it is unset the API is open (local demo mode).
"""

from __future__ import annotations

import json
import os
import secrets
import sqlite3
from pathlib import Path
from typing import Any

from fastapi import Body, Depends, FastAPI, Header, HTTPException, Query
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel

from .agent import run_triage
from .config import REPO_ROOT, SETTINGS
from .models import TriageResult
from .tools.actions import IncidentStore
from .tracing import Tracer

app = FastAPI(
    title="Affiliate KPI triage agent",
    version="0.2.0",
    description="Triage a KPI drop alert: investigate, decide with rules, open an incident, draft an alert. "
    "Read-only towards campaigns. Alerts are drafts that wait for human approval.",
)
_STATIC = Path(__file__).parent / "static" / "index.html"

# How a triage outcome maps to an HTTP status. The body is a TriageResult in every case.
_REJECTION_STATUS = {"invalid_input": 422, "unknown_campaign": 404}


class Incident(BaseModel):
    incident_id: str
    campaign_id: str
    root_cause: str
    severity: str
    status: str
    occurrences: int
    summary: str
    last_alert_id: str
    created_at: str
    updated_at: str


class Health(BaseModel):
    status: str
    llm_mode: str
    page_probe: str
    campaigns: int
    incident_store: str
    auth_required: bool


def require_api_key(x_api_key: str | None = Header(default=None)) -> None:
    """Checked on every data endpoint. The key is read per request so it can be rotated without a restart."""
    expected = os.getenv("TRIAGE_API_KEY", "")
    if expected and not (x_api_key and secrets.compare_digest(x_api_key, expected)):
        raise HTTPException(status_code=401, detail="missing or invalid X-API-Key header")


_EXAMPLE = {
    "alert_id": "ALT-1001",
    "campaign_id": "CMP-101",
    "metric": "conversion_rate",
    "observed_at": "2026-11-16T12:00:00Z",
    "note": "Conversions stopped around 10:00 UTC.",
}


@app.post(
    "/triage",
    response_model=TriageResult,
    dependencies=[Depends(require_api_key)],
    responses={
        422: {"model": TriageResult, "description": "Alert failed validation (status: rejected)"},
        404: {"model": TriageResult, "description": "Campaign does not exist (status: rejected)"},
        500: {"model": TriageResult, "description": "Internal error (status: failed)"},
        401: {"description": "Missing or invalid API key"},
    },
)
def triage(alert: Any = Body(..., examples=[_EXAMPLE])) -> JSONResponse:
    """Run one alert through the workflow. The body is validated inside the workflow (strict
    schema, unknown fields rejected), so every caller gets the same per-field error list."""
    result = run_triage(alert, tracer=Tracer(SETTINGS.runs_dir))
    if result.status == "completed":
        status = 200
    elif result.status == "rejected":
        status = _REJECTION_STATUS.get((result.rejection or {}).get("code", ""), 422)
    else:
        status = 500
    return JSONResponse(status_code=status, content=result.model_dump(mode="json"))


@app.get("/incidents", response_model=list[Incident], dependencies=[Depends(require_api_key)])
def incidents(status: str | None = Query(default=None, pattern="^(open|closed)$")) -> list[dict[str, Any]]:
    return IncidentStore(SETTINGS.db_path).list(status)


@app.get("/samples", dependencies=[Depends(require_api_key)])
def samples() -> dict[str, Any]:
    out: dict[str, Any] = {}
    for path in sorted((REPO_ROOT / "samples" / "alerts").glob("*.json")):
        try:
            out[path.stem] = json.loads(path.read_text())
        except json.JSONDecodeError:
            continue
    return out


@app.get("/health", response_model=Health)
def health() -> JSONResponse:
    """Liveness plus the two local dependencies: the campaign data and the incident store."""
    problems = []
    campaigns = 0
    try:
        campaigns = len(json.loads(SETTINGS.world_path.read_text())["campaigns"])
    except (OSError, ValueError, KeyError) as exc:
        problems.append(f"campaign data: {type(exc).__name__}")
    try:
        IncidentStore(SETTINGS.db_path).list("open")
        store = "ok"
    except (OSError, sqlite3.Error) as exc:
        store = f"error: {type(exc).__name__}"
        problems.append("incident store")
    body = Health(
        status="degraded" if problems else "ok",
        llm_mode=os.getenv("TRIAGE_LLM", "offline"),
        page_probe=SETTINGS.page_probe,
        campaigns=campaigns,
        incident_store=store,
        auth_required=bool(os.getenv("TRIAGE_API_KEY")),
    )
    return JSONResponse(status_code=503 if problems else 200, content=body.model_dump())


@app.get("/", response_class=HTMLResponse, include_in_schema=False)
def index() -> str:
    return _STATIC.read_text()
