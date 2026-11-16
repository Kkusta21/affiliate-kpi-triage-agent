"""HTTP API: status codes, one response schema, health, and the optional API key."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

import triage_agent.api as api
from triage_agent import run_triage
from triage_agent.config import SETTINGS
from triage_agent.models import TriageResult

from conftest import load_alert


@pytest.fixture
def client(tmp_path, monkeypatch):
    settings = SETTINGS.__class__(db_path=str(tmp_path / "incidents.db"), runs_dir=tmp_path / "runs")
    monkeypatch.setattr(api, "SETTINGS", settings)
    monkeypatch.setattr(
        api, "run_triage", lambda raw, tracer: run_triage(raw, tracer=tracer, settings=settings, sleep=lambda _: None)
    )
    monkeypatch.delenv("TRIAGE_API_KEY", raising=False)
    return TestClient(api.app)


def test_completed_triage_is_200_with_the_full_result(client):
    response = client.post("/triage", json=load_alert("01_landing_page_down.json"))
    assert response.status_code == 200
    result = TriageResult.model_validate(response.json())
    assert result.status == "completed" and result.root_cause.value == "landing_page_down"
    assert result.alert_draft.status == "draft_pending_approval" and result.findings[0].primary


@pytest.mark.parametrize(
    "body, status, code",
    [
        ({"alert_id": "nope"}, 422, "invalid_input"),
        (load_alert("10_invalid_input.json"), 422, "invalid_input"),
        ("just a string", 422, "invalid_input"),
        ([1, 2, 3], 422, "invalid_input"),
        (load_alert("11_unknown_campaign.json"), 404, "unknown_campaign"),
    ],
)
def test_rejected_input_has_an_error_status_and_the_same_response_schema(client, body, status, code):
    response = client.post("/triage", json=body)
    assert response.status_code == status
    result = TriageResult.model_validate(response.json())
    assert result.status == "rejected" and result.rejection["code"] == code and result.steps == []


def test_body_that_is_not_json_is_a_422_from_the_framework(client):
    response = client.post("/triage", content="{not json", headers={"Content-Type": "application/json"})
    assert response.status_code == 422


def test_internal_failure_is_500_with_a_structured_body(client, monkeypatch):
    def broken(raw, tracer):
        return TriageResult(
            run_id="r",
            status="failed",
            llm_mode="offline",
            summary="Triage failed",
            rejection={"code": "internal_error", "message": "boom", "errors": []},
        )

    monkeypatch.setattr(api, "run_triage", broken)
    response = client.post("/triage", json=load_alert("01_landing_page_down.json"))
    assert response.status_code == 500 and response.json()["status"] == "failed"


def test_incidents_endpoint_lists_and_filters(client):
    client.post("/triage", json=load_alert("01_landing_page_down.json"))
    client.post("/triage", json=load_alert("01_landing_page_down.json"))
    incidents = client.get("/incidents").json()
    assert len(incidents) == 1 and incidents[0]["occurrences"] == 2 and incidents[0]["status"] == "open"
    assert client.get("/incidents", params={"status": "closed"}).json() == []
    assert client.get("/incidents", params={"status": "anything"}).status_code == 422


def test_health_reports_dependencies_without_auth(client, monkeypatch):
    monkeypatch.setenv("TRIAGE_API_KEY", "s3cret-key")
    body = client.get("/health")
    assert body.status_code == 200
    assert body.json() == {
        "status": "ok",
        "llm_mode": "offline",
        "page_probe": "fixture",
        "campaigns": 12,
        "incident_store": "ok",
        "auth_required": True,
    }


def test_health_is_503_when_the_campaign_data_is_missing(client, tmp_path, monkeypatch):
    monkeypatch.setattr(
        api, "SETTINGS", SETTINGS.__class__(world_path=tmp_path / "missing.json", db_path=str(tmp_path / "i.db"))
    )
    response = client.get("/health")
    assert response.status_code == 503 and response.json()["status"] == "degraded"


def test_api_is_open_when_no_key_is_configured(client):
    assert client.get("/samples").status_code == 200 and client.get("/incidents").status_code == 200


@pytest.mark.parametrize("method, path", [("post", "/triage"), ("get", "/incidents"), ("get", "/samples")])
def test_data_endpoints_require_the_key_when_one_is_configured(client, monkeypatch, method, path):
    monkeypatch.setenv("TRIAGE_API_KEY", "s3cret-key")
    kwargs = {"json": load_alert("06_no_anomaly_low_volume.json")} if method == "post" else {}
    assert getattr(client, method)(path, **kwargs).status_code == 401
    assert getattr(client, method)(path, headers={"X-API-Key": "wrong"}, **kwargs).status_code == 401
    assert getattr(client, method)(path, headers={"X-API-Key": "s3cret-key"}, **kwargs).status_code == 200


def test_unauthorized_request_runs_nothing(client, monkeypatch):
    monkeypatch.setenv("TRIAGE_API_KEY", "s3cret-key")
    client.post("/triage", json=load_alert("01_landing_page_down.json"))
    assert client.get("/incidents", headers={"X-API-Key": "s3cret-key"}).json() == []


def test_openapi_documents_the_error_responses_and_the_ui_loads(client):
    schema = client.get("/openapi.json").json()
    assert set(schema["paths"]["/triage"]["post"]["responses"]) >= {"200", "401", "404", "422", "500"}
    assert "/health" in schema["paths"] and "Run triage" in client.get("/").text
