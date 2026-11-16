"""Tool registry. The planner can only pick names from INVESTIGATION_TOOLS."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from .investigation import (
    ToolContext,
    kpi_snapshot,
    landing_page_check,
    offer_status,
    runbook_search,
    tracking_check,
    traffic_quality,
)


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    fn: Callable[[ToolContext], dict[str, Any]]


INVESTIGATION_TOOLS: dict[str, ToolSpec] = {
    spec.name: spec
    for spec in [
        ToolSpec(
            "kpi_snapshot",
            "Compare the last 3 hours of clicks, conversion rate, revenue and EPC "
            "with the 21 hour baseline. Always the first step.",
            kpi_snapshot,
        ),
        ToolSpec(
            "landing_page_check",
            "Probe the campaign landing page: HTTP status, latency, tracking pixel present.",
            landing_page_check,
        ),
        ToolSpec(
            "offer_status", "Offer state at the advertiser: active or paused, daily cap, allowed countries.", offer_status
        ),
        ToolSpec("tracking_check", "Tracker funnel: clicks recorded, pixel events, postbacks received.", tracking_check),
        ToolSpec(
            "traffic_quality",
            "Traffic by sub source and country: bot scores, share of disallowed countries.",
            traffic_quality,
        ),
        ToolSpec(
            "runbook_search", "Search the internal runbooks for the handling procedure. Needs a short query.", runbook_search
        ),
    ]
}

FINISH = "finish"

__all__ = ["INVESTIGATION_TOOLS", "FINISH", "ToolContext", "ToolSpec"]
