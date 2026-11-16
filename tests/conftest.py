from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from triage_agent.agent import run_triage
from triage_agent.datasource import DataSource
from triage_agent.llm import OfflineLLM
from triage_agent.tools.actions import IncidentStore
from triage_agent.tracing import Tracer

ROOT = Path(__file__).resolve().parents[1]
ALERTS = ROOT / "samples" / "alerts"


def alert(campaign_id: str = "CMP-101", **overrides: Any) -> dict[str, Any]:
    base = {"alert_id": "ALT-1", "campaign_id": campaign_id, "metric": "conversion_rate",
            "observed_at": "2026-11-16T12:00:00Z"}
    return {**base, **overrides}


def load_alert(name: str) -> Any:
    return json.loads((ALERTS / name).read_text())


@pytest.fixture
def store() -> IncidentStore:
    return IncidentStore(":memory:")


@pytest.fixture
def triage(store):
    """Run the full workflow with no sleeping, no files and an in-memory incident store."""
    def _run(raw: Any, llm=None, **kwargs):
        tracer = Tracer(None)
        sleeps: list[float] = []
        result = run_triage(raw, llm=llm or OfflineLLM(), ds=kwargs.pop("ds", None) or DataSource(),
                            incidents=store, tracer=tracer, sleep=sleeps.append, **kwargs)
        result.__dict__["_trace"] = tracer.events
        result.__dict__["_sleeps"] = sleeps
        return result
    return _run
