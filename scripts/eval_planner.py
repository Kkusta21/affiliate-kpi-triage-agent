"""Compare planner modes on every sample alert: the offline policy vs the model in TRIAGE_LLM.

    python scripts/eval_planner.py

Each sample runs with a fresh in-memory incident store, no trace files and no retry sleeps.
"Verdict agreement" compares status, root cause and severity with samples/expected/.
The live column is skipped when TRIAGE_LLM is unset or "offline", or when its API key is missing.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

from triage_agent.agent import run_triage
from triage_agent.llm import make_llm
from triage_agent.tools.actions import IncidentStore
from triage_agent.tracing import Tracer

ROOT = Path(__file__).resolve().parents[1]
ALERTS = sorted((ROOT / "samples" / "alerts").glob("*.json"))
KEY_ENV = {"openai": "OPENAI_API_KEY", "anthropic": "ANTHROPIC_API_KEY", "google_genai": "GOOGLE_API_KEY"}


def verdict_of(result: dict) -> tuple:
    return result["status"], result.get("root_cause"), result.get("severity")


def run_mode(spec: str) -> dict:
    agree, tool_calls, fallbacks, hyp_agree, completed, misses = 0, 0, 0, 0, 0, []
    llm_failures = 0
    started = time.perf_counter()
    for path in ALERTS:
        expected = json.loads((ROOT / "samples" / "expected" / path.name).read_text())
        result = run_triage(
            json.loads(path.read_text()),
            llm=make_llm(spec),
            incidents=IncidentStore(":memory:"),
            tracer=Tracer(None),
            sleep=lambda _: None,
        ).model_dump(mode="json")
        if verdict_of(result) == verdict_of(expected):
            agree += 1
        else:
            misses.append(f"{path.stem}: got {verdict_of(result)}, expected {verdict_of(expected)}")
        if result["status"] == "completed":
            completed += 1
            tool_calls += len(result["steps"])
            fallbacks += "planner_fallback_used" in result["guardrail_flags"]
            hyp_agree += result["llm_hypothesis"] == result["root_cause"]
            if spec != "offline" and result["llm_hypothesis"] != result["root_cause"]:
                misses.append(
                    f"{path.stem}: model hypothesis {result['llm_hypothesis']!r}, "
                    f"rules verdict {result['root_cause']!r} (verdict unchanged, run flagged for review)"
                )
            llm_failures += any(
                flag in result["guardrail_flags"] for flag in ("hypothesis_unavailable", "analyst_note_unavailable")
            )
    seconds = time.perf_counter() - started
    return {
        "Verdict agreement with expected": f"{agree}/{len(ALERTS)}",
        "Avg tool calls per completed run": f"{tool_calls / max(completed, 1):.2f}",
        "Planner fallback rate": f"{fallbacks / max(completed, 1):.0%}",
        "Model hypothesis = rules verdict": "n/a" if spec == "offline" else f"{hyp_agree}/{completed}",
        "Runs with a failed hypothesis or note call": "n/a" if spec == "offline" else f"{llm_failures}/{completed}",
        "Wall time, all samples": f"{seconds:.1f} s",
        "_misses": misses,
    }


def live_spec() -> str | None:
    spec = os.getenv("TRIAGE_LLM", "offline").strip()
    if spec == "offline":
        print("TRIAGE_LLM is not set to a model, live column skipped.\n")
        return None
    key = KEY_ENV.get(spec.split(":", 1)[0])
    if key and not os.getenv(key):
        print(f"{key} is not set, live column for {spec} skipped.\n")
        return None
    return spec


def main() -> None:
    modes = {"offline": run_mode("offline")}
    spec = live_spec()
    if spec:
        modes[spec] = run_mode(spec)
    header = "| Metric | " + " | ".join(modes) + " |"
    print(header)
    print("|" + "---|" * (len(modes) + 1))
    for metric in next(iter(modes.values())):
        if not metric.startswith("_"):
            print(f"| {metric} | " + " | ".join(m[metric] for m in modes.values()) + " |")
    for name, mode in modes.items():
        for miss in mode["_misses"]:
            print(f"\n[{name}] note: {miss}")


if __name__ == "__main__":
    main()
