"""Regenerate samples/expected/*.json from the sample alerts (offline planner, fresh store each)."""
from __future__ import annotations

import json
from pathlib import Path

from triage_agent.agent import run_triage
from triage_agent.llm import OfflineLLM
from triage_agent.tools.actions import IncidentStore
from triage_agent.tracing import Tracer

ROOT = Path(__file__).resolve().parents[1]


def normalise(result: dict) -> dict:
    """Drop the fields that change on every run."""
    result = dict(result)
    result.pop("run_id", None)
    result["steps"] = [{k: v for k, v in step.items() if k != "duration_ms"} for step in result["steps"]]
    return result


def expected_for(alert_path: Path) -> dict:
    raw = json.loads(alert_path.read_text())
    result = run_triage(raw, llm=OfflineLLM(), incidents=IncidentStore(":memory:"), tracer=Tracer(None),
                        sleep=lambda _: None)
    return normalise(result.model_dump(mode="json"))


if __name__ == "__main__":
    out_dir = ROOT / "samples" / "expected"
    out_dir.mkdir(exist_ok=True)
    for path in sorted((ROOT / "samples" / "alerts").glob("*.json")):
        (out_dir / path.name).write_text(json.dumps(expected_for(path), indent=2) + "\n")
        print("wrote", path.name)
