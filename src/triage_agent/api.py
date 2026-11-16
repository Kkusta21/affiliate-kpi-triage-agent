"""Small HTTP API and a one page UI on top of the same workflow.

    uvicorn triage_agent.api:app --port 8000
    POST /triage      body: alert JSON            -> TriageResult
    GET  /incidents                                -> incident list
    GET  /samples                                  -> sample alerts for the UI
    GET  /                                         -> demo page
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from fastapi import Body, FastAPI
from fastapi.responses import HTMLResponse

from .agent import run_triage
from .config import REPO_ROOT, SETTINGS
from .models import TriageResult
from .tools.actions import IncidentStore
from .tracing import Tracer

app = FastAPI(title="Affiliate KPI triage agent", version="0.1.0")
_STATIC = Path(__file__).parent / "static" / "index.html"


@app.post("/triage", response_model=TriageResult)
def triage(alert: Any = Body(...)) -> TriageResult:
    # Validation happens inside the workflow so API, CLI and tests share one rejection path.
    return run_triage(alert, tracer=Tracer(SETTINGS.runs_dir))


@app.get("/incidents")
def incidents(status: str | None = None) -> list[dict[str, Any]]:
    return IncidentStore(SETTINGS.db_path).list(status)


@app.get("/samples")
def samples() -> dict[str, Any]:
    out: dict[str, Any] = {}
    for path in sorted((REPO_ROOT / "samples" / "alerts").glob("*.json")):
        try:
            out[path.stem] = json.loads(path.read_text())
        except json.JSONDecodeError:
            continue
    return out


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return _STATIC.read_text()
