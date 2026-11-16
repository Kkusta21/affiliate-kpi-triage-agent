"""Graded findings: confirmed, suspected and inconclusive, several at once, never duplicated."""
from __future__ import annotations

from triage_agent import rules
from triage_agent.config import SETTINGS
from triage_agent.models import FindingStatus, RootCause, Severity

from conftest import load_alert
from test_rules import OFFER, PAGE, TRACKING, TRAFFIC, full

BOTS = {**TRAFFIC, "worst_source": {"sub_id": "s9", "share": 0.6, "bot_score": 0.9}}


def by_cause(verdict):
    return {f.cause.value: f for f in verdict.findings}


def test_primary_finding_carries_metrics_explanation_and_actions():
    verdict = rules.evaluate(full(traffic_quality=BOTS), SETTINGS)
    primary = verdict.findings[0]
    assert primary.primary and primary.cause is RootCause.traffic_quality
    assert primary.status is FindingStatus.confirmed and primary.evidence_strength == "strong"
    assert primary.supporting_metrics == {"sub_id": "s9", "share": 0.6, "bot_score": 0.9}
    assert primary.severity is verdict.severity is Severity.high
    assert primary.recommended_actions == verdict.recommended_actions and "s9" in primary.explanation
    assert verdict.stop_reason == "confirmed by traffic_quality"


def test_two_causes_in_one_check_are_both_reported_once():
    traffic = {**BOTS, "disallowed_geo_share": 0.55, "disallowed_geos": ["RO"]}
    verdict = rules.evaluate(full(traffic_quality=traffic), SETTINGS)
    assert verdict.root_cause is RootCause.traffic_quality
    assert [(f.cause.value, f.status.value, f.primary) for f in verdict.findings] == [
        ("traffic_quality", "confirmed", True), ("geo_mismatch", "confirmed", False)]
    assert verdict.other_findings[0].startswith("geo_mismatch: 55% of traffic")


def test_every_rule_that_matches_gathered_evidence_is_reported_without_duplicates():
    # A model that gathered everything: paused AND capped offer, broken tracking (two tracking
    # rules match, one finding), bots and wrong countries.
    evidence = full(
        offer_status={**OFFER, "status": "paused", "cap_reached": True},
        landing_page_check={**PAGE, "has_tracking_pixel": False},
        tracking_check={**TRACKING, "pixel_events": 0, "postback_ratio": None},
        traffic_quality={**BOTS, "disallowed_geo_share": 0.55, "disallowed_geos": ["RO"]})
    verdict = rules.evaluate(evidence, SETTINGS)
    causes = [f.cause.value for f in verdict.findings]
    assert verdict.root_cause is RootCause.offer_paused                # first in priority order
    assert causes == ["offer_paused", "offer_cap_reached", "tracking_broken", "traffic_quality", "geo_mismatch"]
    assert len(causes) == len(set(causes)) and sum(f.primary for f in verdict.findings) == 1


def test_no_tool_is_requested_once_the_drop_is_fully_explained():
    evidence = {"kpi_snapshot": full()["kpi_snapshot"], "landing_page_check": PAGE,
                "offer_status": {**OFFER, "cap_reached": True}}
    assert rules.next_needed(evidence, SETTINGS) is None
    assert [f.cause.value for f in rules.evaluate(evidence, SETTINGS).findings] == ["offer_cap_reached"]


def test_weak_postback_ratio_explains_only_part_of_the_drop_so_traffic_is_still_required():
    evidence = full(tracking_check={**TRACKING, "postbacks_received": 4, "postback_ratio": 0.2})
    partial = {k: v for k, v in evidence.items() if k != "traffic_quality"}
    assert rules.next_needed(partial, SETTINGS) == "traffic_quality"
    verdict = rules.evaluate(full(tracking_check=evidence["tracking_check"], traffic_quality=BOTS), SETTINGS)
    assert verdict.root_cause is RootCause.tracking_broken
    assert [f.cause.value for f in verdict.findings] == ["tracking_broken", "traffic_quality"]


def test_near_threshold_values_are_suspected_and_never_the_root_cause():
    near = full(traffic_quality={**TRAFFIC, "worst_source": {"sub_id": "s9", "share": 0.5, "bot_score": 0.6},
                                 "disallowed_geo_share": 0.26, "disallowed_geos": ["RO"]},
                landing_page_check={**PAGE, "latency_ms": 9000},
                tracking_check={**TRACKING, "postbacks_received": 11, "postback_ratio": 0.55},
                offer_status={**OFFER, "cap_used_share": 0.9})
    verdict = rules.evaluate(near, SETTINGS)
    assert verdict.root_cause is RootCause.unknown and verdict.needs_human_review
    assert {f.cause.value: f.status.value for f in verdict.findings} == {
        "landing_page_down": "suspected", "offer_cap_reached": "suspected", "tracking_broken": "suspected",
        "traffic_quality": "suspected", "geo_mismatch": "suspected"}
    assert not any(f.primary or f.recommended_actions for f in verdict.findings)
    assert any(r.startswith("Suspected but unconfirmed: ") for r in verdict.reasons)
    assert "traffic_quality (suspected): " in " | ".join(verdict.other_findings)


def test_a_suspected_finding_does_not_stop_the_investigation():
    evidence = {"kpi_snapshot": full()["kpi_snapshot"], "landing_page_check": {**PAGE, "latency_ms": 9000}}
    assert rules.next_needed(evidence, SETTINGS) == "offer_status"


def test_failed_check_is_inconclusive_for_the_causes_it_would_have_tested():
    verdict = rules.evaluate(full(tracking_check={"error": "502"}), SETTINGS)
    finding = by_cause(verdict)["tracking_broken"]
    assert finding.status is FindingStatus.inconclusive and not finding.primary
    assert verdict.root_cause is RootCause.unknown and verdict.unavailable_checks == ["tracking_check"]
    failed_offer = rules.evaluate(full(offer_status={"error": "timeout"}, traffic_quality=BOTS), SETTINGS)
    statuses = {f.cause.value: f.status.value for f in failed_offer.findings}
    assert statuses == {"offer_paused": "inconclusive", "offer_cap_reached": "inconclusive",
                        "traffic_quality": "confirmed"}
    assert failed_offer.root_cause is RootCause.traffic_quality and failed_offer.confidence == 0.7


def test_confirmed_finding_wins_over_inconclusive_for_the_same_cause():
    # The tracker is down, but the page itself shows the pixel is gone: that is enough.
    evidence = full(landing_page_check={**PAGE, "has_tracking_pixel": False}, tracking_check={"error": "502"})
    verdict = rules.evaluate(evidence, SETTINGS)
    assert verdict.root_cause is RootCause.tracking_broken
    assert [(f.cause.value, f.status.value) for f in verdict.findings] == [("tracking_broken", "confirmed")]


def test_http_404_is_a_broken_page_not_a_tracking_problem():
    page = {**PAGE, "status_code": 404, "outcome": "client_error", "reachable": False, "has_tracking_pixel": False}
    verdict = rules.evaluate({"kpi_snapshot": full()["kpi_snapshot"], "landing_page_check": page}, SETTINGS)
    assert verdict.root_cause is RootCause.landing_page_down and "HTTP 404 (client error)" in verdict.reasons[0]
    assert [f.cause.value for f in verdict.findings] == ["landing_page_down"]   # the missing pixel is not blamed


def test_no_anomaly_and_insufficient_data_carry_no_findings():
    quiet = {"kpi_snapshot": {**full()["kpi_snapshot"], "significant": False, "enough_volume": False}}
    assert rules.evaluate(quiet, SETTINGS).findings == []
    assert rules.evaluate(quiet, SETTINGS).stop_reason == "decided from the KPI snapshot alone"


def test_weak_kpi_signal_lowers_confidence_and_asks_for_review():
    kpi = {**full()["kpi_snapshot"], "signal_strength": "weak"}
    verdict = rules.evaluate(full(kpi_snapshot=kpi, traffic_quality=BOTS), SETTINGS)
    assert verdict.root_cause is RootCause.traffic_quality
    assert verdict.confidence == 0.7 and verdict.needs_human_review


def test_end_to_end_results_expose_findings_and_stop_reason(triage):
    both = triage(load_alert("12_slow_page_and_offer_cap.json"))
    assert [(f.cause.value, f.status.value, f.primary) for f in both.findings] == [
        ("landing_page_down", "confirmed", True), ("offer_cap_reached", "confirmed", False)]
    assert both.findings[0].evidence_strength == "moderate" and both.stop_reason == "confirmed by landing_page_check"
    outage = triage(load_alert("08_tracker_api_down.json"))
    assert [(f.cause.value, f.status.value) for f in outage.findings] == [("tracking_broken", "inconclusive")]
    not_found = triage(load_alert("14_landing_page_404.json"))
    assert not_found.root_cause.value == "landing_page_down" and len(not_found.steps) == 3
    assert not_found.evidence["landing_page_check"]["outcome"] == "client_error"
    sparse = triage(load_alert("15_insufficient_kpi_data.json"))
    assert sparse.root_cause.value == "unknown" and sparse.needs_human_review and sparse.findings == []
    assert sparse.steps[0].tool == "kpi_snapshot" and "landing_page_check" not in [s.tool for s in sparse.steps]
