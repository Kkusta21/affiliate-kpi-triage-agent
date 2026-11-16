"""The verdict logic, tested on hand-built evidence with no agent, model or I/O involved."""

import pytest

from triage_agent import rules
from triage_agent.config import SETTINGS
from triage_agent.models import RootCause, Severity

KPI = {
    "metric": "conversion_rate",
    "metric_change_pct": -60.0,
    "enough_volume": True,
    "significant": True,
    "est_revenue_loss_per_hour_usd": 120.0,
    "current": {"clicks_total": 900},
}
PAGE = {"status_code": 200, "latency_ms": 200, "reachable": True, "slow": False, "has_tracking_pixel": True}
OFFER = {"offer_id": "OFF-1", "status": "active", "cap_reached": False, "daily_cap": 100, "conversions_today": 10}
TRACKING = {"clicks_recorded": 900, "pixel_events": 20, "postbacks_received": 20, "postback_ratio": 1.0}
TRAFFIC = {
    "worst_source": {"sub_id": "s1", "share": 0.5, "bot_score": 0.05},
    "disallowed_geo_share": 0.02,
    "disallowed_geos": ["CH"],
}


def full(**overrides):
    evidence = {
        "kpi_snapshot": KPI,
        "landing_page_check": PAGE,
        "offer_status": OFFER,
        "tracking_check": TRACKING,
        "traffic_quality": TRAFFIC,
    }
    evidence.update(overrides)
    return evidence


def test_checks_are_requested_in_priority_order():
    evidence = {}
    order = []
    for tool, data in [
        ("kpi_snapshot", KPI),
        ("landing_page_check", PAGE),
        ("offer_status", OFFER),
        ("tracking_check", TRACKING),
        ("traffic_quality", TRAFFIC),
    ]:
        order.append(rules.next_needed(evidence, SETTINGS))
        evidence[tool] = data
    assert order == rules.CHECK_ORDER
    assert rules.next_needed(evidence, SETTINGS) is None


def test_no_anomaly_stops_after_kpi():
    quiet = {"kpi_snapshot": {**KPI, "significant": False, "enough_volume": False}}
    assert rules.next_needed(quiet, SETTINGS) is None
    verdict = rules.evaluate(quiet, SETTINGS)
    assert verdict.root_cause is RootCause.no_anomaly and verdict.severity is Severity.info
    assert verdict.est_revenue_loss_per_hour_usd == 0 and not verdict.needs_human_review


@pytest.mark.parametrize(
    "overrides, cause, severity",
    [
        (
            {"landing_page_check": {**PAGE, "status_code": 503, "reachable": False}},
            RootCause.landing_page_down,
            Severity.critical,
        ),
        (
            {"landing_page_check": {**PAGE, "slow": True, "latency_ms": 20000}},
            RootCause.landing_page_down,
            Severity.critical,
        ),
        ({"offer_status": {**OFFER, "status": "paused"}}, RootCause.offer_paused, Severity.medium),
        ({"offer_status": {**OFFER, "cap_reached": True}}, RootCause.offer_cap_reached, Severity.medium),
        ({"landing_page_check": {**PAGE, "has_tracking_pixel": False}}, RootCause.tracking_broken, Severity.high),
        (
            {"tracking_check": {**TRACKING, "pixel_events": 0, "postback_ratio": None}},
            RootCause.tracking_broken,
            Severity.high,
        ),
        (
            {"tracking_check": {**TRACKING, "postbacks_received": 4, "postback_ratio": 0.2}},
            RootCause.tracking_broken,
            Severity.high,
        ),
        (
            {"traffic_quality": {**TRAFFIC, "worst_source": {"sub_id": "s9", "share": 0.6, "bot_score": 0.9}}},
            RootCause.traffic_quality,
            Severity.high,
        ),
        (
            {"traffic_quality": {**TRAFFIC, "disallowed_geo_share": 0.55, "disallowed_geos": ["RO"]}},
            RootCause.geo_mismatch,
            Severity.high,
        ),
        ({}, RootCause.unknown, Severity.high),
    ],
)
def test_each_root_cause(overrides, cause, severity):
    verdict = rules.evaluate(full(**overrides), SETTINGS)
    assert verdict.root_cause is cause
    assert verdict.severity is severity
    assert verdict.needs_human_review == (cause is RootCause.unknown)


def test_small_bot_source_is_not_blamed():
    evidence = full(traffic_quality={**TRAFFIC, "worst_source": {"sub_id": "s9", "share": 0.05, "bot_score": 0.95}})
    assert rules.evaluate(evidence, SETTINGS).root_cause is RootCause.unknown


def test_failed_check_lowers_confidence_and_asks_for_review():
    evidence = full(
        tracking_check={"error": "502"}, traffic_quality={**TRAFFIC, "disallowed_geo_share": 0.6, "disallowed_geos": ["RO"]}
    )
    verdict = rules.evaluate(evidence, SETTINGS)
    assert verdict.root_cause is RootCause.geo_mismatch
    assert verdict.confidence == 0.7 and verdict.needs_human_review
    assert verdict.unavailable_checks == ["tracking_check"]


def test_model_disagreement_is_flagged_but_rules_win():
    verdict = rules.evaluate(
        full(landing_page_check={**PAGE, "status_code": 503, "reachable": False}), SETTINGS, llm_hypothesis="no_anomaly"
    )
    assert verdict.root_cause is RootCause.landing_page_down
    assert verdict.needs_human_review and verdict.confidence == 0.7


SLOW_PAGE = {**PAGE, "slow": True, "latency_ms": 15000}


def test_slow_page_keeps_checking_and_reports_the_second_cause():
    assert rules.next_needed({"kpi_snapshot": KPI, "landing_page_check": SLOW_PAGE}, SETTINGS) == "offer_status"
    verdict = rules.evaluate(full(landing_page_check=SLOW_PAGE, offer_status={**OFFER, "cap_reached": True}), SETTINGS)
    assert verdict.root_cause is RootCause.landing_page_down and verdict.severity is Severity.critical
    assert len(verdict.other_findings) == 1 and verdict.other_findings[0].startswith("offer_cap_reached: ")
    assert verdict.confidence == 0.9 and not verdict.needs_human_review


def test_dead_page_still_stops_at_once_with_no_other_findings():
    evidence = {"kpi_snapshot": KPI, "landing_page_check": {**PAGE, "status_code": 503, "reachable": False}}
    assert rules.next_needed(evidence, SETTINGS) is None
    assert rules.evaluate(evidence, SETTINGS).other_findings == []


def test_slow_page_cut_short_by_the_budget_is_kept_but_sent_for_review():
    verdict = rules.evaluate({"kpi_snapshot": KPI, "landing_page_check": SLOW_PAGE}, SETTINGS)
    assert verdict.root_cause is RootCause.landing_page_down
    assert verdict.needs_human_review and verdict.confidence == 0.7
    assert "Investigation stopped before 'offer_status' could run." in verdict.reasons


def test_incomplete_evidence_never_produces_a_guess():
    verdict = rules.evaluate({"kpi_snapshot": KPI}, SETTINGS)
    assert verdict.root_cause is RootCause.unknown and verdict.needs_human_review
