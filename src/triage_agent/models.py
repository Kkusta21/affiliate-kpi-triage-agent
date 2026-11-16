"""Typed contracts: the alert that comes in and the triage result that goes out."""
from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class Metric(str, Enum):
    conversion_rate = "conversion_rate"
    revenue = "revenue"
    epc = "epc"
    clicks = "clicks"


class RootCause(str, Enum):
    landing_page_down = "landing_page_down"
    offer_paused = "offer_paused"
    offer_cap_reached = "offer_cap_reached"
    tracking_broken = "tracking_broken"
    traffic_quality = "traffic_quality"
    geo_mismatch = "geo_mismatch"
    no_anomaly = "no_anomaly"
    unknown = "unknown"


class Severity(str, Enum):
    critical = "critical"
    high = "high"
    medium = "medium"
    low = "low"
    info = "info"


class Alert(BaseModel):
    """Input. Comes from a KPI monitor or is typed by an affiliate manager."""

    model_config = ConfigDict(extra="forbid")

    alert_id: str = Field(pattern=r"^ALT-\d{1,8}$")
    campaign_id: str = Field(pattern=r"^CMP-\d{1,8}$")
    metric: Metric
    observed_at: datetime
    note: str | None = Field(default=None, max_length=2000)


class PlanStep(BaseModel):
    """What the planner is allowed to say. Arguments are filled in by code, not by the model."""

    model_config = ConfigDict(extra="forbid")

    action: str
    reason: str = Field(default="", max_length=300)
    runbook_query: str | None = Field(default=None, max_length=120)


class Verdict(BaseModel):
    root_cause: RootCause
    severity: Severity
    confidence: float = Field(ge=0, le=1)
    needs_human_review: bool
    reasons: list[str]
    recommended_actions: list[str]
    owner_role: str | None = None
    est_revenue_loss_per_hour_usd: float = 0.0
    unavailable_checks: list[str] = Field(default_factory=list)


class StepRecord(BaseModel):
    step: int
    tool: str
    status: Literal["ok", "failed"]
    attempts: int
    duration_ms: float
    chosen_by: str
    reason: str = ""


class AlertDraft(BaseModel):
    to_role: str
    channel: str
    subject: str
    body: str
    status: Literal["draft_pending_approval"] = "draft_pending_approval"


class IncidentRef(BaseModel):
    action: Literal["created", "updated", "none"]
    incident_id: str | None = None
    occurrences: int = 0


class TriageResult(BaseModel):
    """Structured output. `summary` is the human-readable version of the same decision."""

    run_id: str
    status: Literal["completed", "rejected", "failed"]
    alert_id: str | None = None
    campaign_id: str | None = None
    campaign_name: str | None = None
    llm_mode: str
    root_cause: RootCause | None = None
    severity: Severity | None = None
    confidence: float | None = None
    needs_human_review: bool = False
    reasons: list[str] = Field(default_factory=list)
    recommended_actions: list[str] = Field(default_factory=list)
    est_revenue_loss_per_hour_usd: float = 0.0
    llm_hypothesis: str | None = None
    kpi: dict[str, Any] = Field(default_factory=dict)
    evidence: dict[str, Any] = Field(default_factory=dict)
    runbook_refs: list[str] = Field(default_factory=list)
    incident: IncidentRef = Field(default_factory=lambda: IncidentRef(action="none"))
    alert_draft: AlertDraft | None = None
    refused_requests: list[dict[str, str]] = Field(default_factory=list)
    guardrail_flags: list[str] = Field(default_factory=list)
    rejection: dict[str, Any] | None = None
    steps: list[StepRecord] = Field(default_factory=list)
    summary: str = ""
