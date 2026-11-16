"""End to end: sample alert in, full workflow, structured result out."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from triage_agent.models import TriageResult

from conftest import ALERTS, ROOT, alert, load_alert

sys.path.insert(0, str(ROOT / "scripts"))
from make_expected import expected_for  # noqa: E402

SAMPLES = sorted(p.name for p in ALERTS.glob("*.json"))


@pytest.mark.parametrize("name", SAMPLES)
def test_sample_matches_expected_output(name):
    expected = json.loads((ROOT / "samples" / "expected" / name).read_text())
    assert expected_for(ALERTS / name) == expected


@pytest.mark.parametrize("name, cause, owner, steps", [
    ("01_landing_page_down.json", "landing_page_down", "tech_oncall", 3),
    ("02_tracking_broken_with_injection_on_page.json", "tracking_broken", "tracking_team", 5),
    ("03_bot_traffic_with_action_request.json", "traffic_quality", "affiliate_manager", 6),
    ("04_offer_cap_reached.json", "offer_cap_reached", "affiliate_manager", 4),
    ("05_geo_mismatch.json", "geo_mismatch", "affiliate_manager", 6),
    ("07_unknown_cause_flaky_probe.json", "unknown", "affiliate_manager", 6),
    ("12_slow_page_and_offer_cap.json", "landing_page_down", "tech_oncall", 4),
])
def test_business_cases(triage, name, cause, owner, steps):
    result = triage(load_alert(name))
    assert result.status == "completed" and result.root_cause.value == cause
    assert result.incident.action == "created" and result.alert_draft.to_role == owner
    assert len(result.steps) == steps and result.steps[0].tool == "kpi_snapshot"
    assert cause in result.summary
    TriageResult.model_validate(json.loads(result.model_dump_json()))  # JSON round trip stays valid


def test_no_anomaly_opens_nothing(triage, store):
    result = triage(load_alert("06_no_anomaly_low_volume.json"))
    assert result.root_cause.value == "no_anomaly" and result.incident.action == "none"
    assert result.alert_draft is None and store.list() == []
    assert [s.tool for s in result.steps] == ["kpi_snapshot"]


def test_two_problems_at_once_reports_both(triage):
    result = triage(load_alert("12_slow_page_and_offer_cap.json"))
    assert result.root_cause.value == "landing_page_down"  # first match in priority order
    assert [f.split(":")[0] for f in result.other_findings] == ["offer_cap_reached"]
    assert "Also found" in result.summary and "Also found: offer_cap_reached" in result.alert_draft.body


def test_flaky_probe_is_retried_with_backoff(triage):
    result = triage(load_alert("07_unknown_cause_flaky_probe.json"))
    probe = next(s for s in result.steps if s.tool == "landing_page_check")
    assert (probe.status, probe.attempts) == ("ok", 3)
    assert result._sleeps == [0.5, 1.0]
    assert result.needs_human_review


def test_tool_outage_degrades_instead_of_guessing(triage):
    result = triage(load_alert("08_tracker_api_down.json"))
    failed = next(s for s in result.steps if s.tool == "tracking_check")
    assert (failed.status, failed.attempts) == ("failed", 3)
    assert result.root_cause.value == "unknown" and result.needs_human_review
    assert "tool_failed:tracking_check" in result.guardrail_flags
    assert "tracking_check" in " ".join(result.reasons)


def test_action_requests_are_refused_but_triage_completes(triage):
    result = triage(load_alert("03_bot_traffic_with_action_request.json"))
    assert [r["request"] for r in result.refused_requests] == ["block sub_778", "pause the campaign"]
    assert "action_request_refused" in result.guardrail_flags and "Refused request" in result.summary
    assert result.alert_draft.status == "draft_pending_approval"


def test_injection_in_note_changes_nothing(triage):
    clean = triage(load_alert("01_landing_page_down.json"))
    attacked = triage(load_alert("09_prompt_injection_in_note.json"))
    assert attacked.root_cause == clean.root_cause and attacked.severity == clean.severity
    assert "prompt_injection_suspected_in_note" in attacked.guardrail_flags
    assert attacked.incident.action == "updated" and attacked.incident.occurrences == 2  # same incident, no duplicate


def test_injection_on_landing_page_changes_nothing(triage):
    result = triage(load_alert("02_tracking_broken_with_injection_on_page.json"))
    assert result.root_cause.value == "tracking_broken"
    assert "prompt_injection_suspected_in:landing_page_check" in result.guardrail_flags
    assert "ignore previous instructions" not in result.model_dump_json().lower()


@pytest.mark.parametrize("raw, code", [
    (load_alert("10_invalid_input.json"), "invalid_input"),
    (load_alert("11_unknown_campaign.json"), "unknown_campaign"),
    ("not json at all", "invalid_input"),
    (None, "invalid_input"),
])
def test_bad_input_is_rejected_without_running_tools(triage, store, raw, code):
    result = triage(raw)
    assert result.status == "rejected" and result.rejection["code"] == code
    assert result.steps == [] and store.list() == []
    assert not any(e["event"] == "tool_call" for e in result._trace)


def test_trace_records_every_decision(triage):
    result = triage(load_alert("04_offer_cap_reached.json"))
    kinds = [e["event"] for e in result._trace]
    assert kinds[0] == "run_start" and kinds[-1] == "final"
    assert kinds.count("tool_call") == len(result.steps) + 2  # investigation tools + incident_store + draft_alert
    assert {"guardrail", "plan", "diagnosis"} <= set(kinds)
    assert all(e["run_id"] == result.run_id for e in result._trace)


def test_trace_file_is_written(tmp_path):
    from triage_agent.agent import run_triage
    from triage_agent.tools.actions import IncidentStore
    from triage_agent.tracing import Tracer

    tracer = Tracer(tmp_path)
    run_triage(alert("CMP-106"), incidents=IncidentStore(":memory:"), tracer=tracer)
    lines = [json.loads(line) for line in tracer.path.read_text().splitlines()]
    assert lines[-1]["event"] == "final" and lines[-1]["root_cause"] == "no_anomaly"


def test_http_api(tmp_path, monkeypatch):
    import triage_agent.api as api
    from triage_agent.config import SETTINGS

    monkeypatch.setattr(api, "SETTINGS", SETTINGS.__class__(db_path=str(tmp_path / "i.db"), runs_dir=tmp_path / "runs"))
    monkeypatch.setattr(api, "run_triage", lambda raw, tracer: __import__("triage_agent").run_triage(
        raw, tracer=tracer, settings=api.SETTINGS, sleep=lambda _: None))
    client = TestClient(api.app)

    ok = client.post("/triage", json=load_alert("01_landing_page_down.json"))
    assert ok.status_code == 200 and ok.json()["root_cause"] == "landing_page_down"
    bad = client.post("/triage", json={"alert_id": "nope"})
    assert bad.status_code == 200 and bad.json()["status"] == "rejected"
    assert client.get("/incidents").json()[0]["incident_id"] == "INC-0001"
    assert "01_landing_page_down" in client.get("/samples").json()
    assert "Run triage" in client.get("/").text
