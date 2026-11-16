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


class FindingStatus(str, Enum):
    confirmed = "confirmed"  # a rule threshold is clearly met on healthy evidence
    suspected = "suspected"  # close to a threshold: worth a look, never the root cause by itself
    inconclusive = "inconclusive"  # the check could not run or the data is incomplete: cannot be ruled out


class Finding(BaseModel):
    """One independently supported finding. Built only by rules.py from tool evidence."""

    cause: RootCause
    status: FindingStatus
    primary: bool = False
    severity: Severity
    evidence_strength: Literal["strong", "moderate", "weak"]
    explanation: str
    supporting_metrics: dict[str, Any] = Field(default_factory=dict)
    recommended_actions: list[str] = Field(default_factory=list)


class Verdict(BaseModel):
    root_cause: RootCause
    severity: Severity
    # A fixed heuristic rank (0.9 clean, 0.7 degraded, 0.4 unknown, minus 0.2 on model disagreement).
    # NOT a calibrated probability: it has never been fitted to how often verdicts are right.
    confidence: float = Field(ge=0, le=1, description="Heuristic diagnostic confidence rank. Not a calibrated probability.")
    needs_human_review: bool
    reasons: list[str]
    recommended_actions: list[str]
    findings: list[Finding] = Field(default_factory=list)  # every finding, the primary one first
    other_findings: list[str] = Field(default_factory=list)  # the non-primary findings as short text
    stop_reason: str = ""  # why the investigation ended where it did
    owner_role: str | None = None
    est_revenue_loss_per_hour_usd: float = 0.0
    unavailable_checks: list[str] = Field(default_factory=list)


class LLMUsage(BaseModel):
    """Totals for the model calls of one run. All zero in offline mode."""

    model: str | None = None
    calls: int = 0
    failed_calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    latency_ms: float = 0.0


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
    findings: list[Finding] = Field(default_factory=list)
    other_findings: list[str] = Field(default_factory=list)
    stop_reason: str = ""
    recommended_actions: list[str] = Field(default_factory=list)
    est_revenue_loss_per_hour_usd: float = 0.0
    llm_usage: LLMUsage = Field(default_factory=LLMUsage)
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
