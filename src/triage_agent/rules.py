"""Deterministic diagnosis.

The model plans the investigation and writes prose. The decision itself is made here,
by plain rules over the collected evidence, so the same evidence always gives the same
verdict and every verdict can be explained line by line.

The checks run in a fixed priority order (cheapest and most damaging first):
    KPI signal -> landing page -> offer -> tracking -> traffic
"""
from __future__ import annotations

from typing import Any

from .config import Settings
from .models import RootCause, Severity, Verdict

CHECK_ORDER = ["kpi_snapshot", "landing_page_check", "offer_status", "tracking_check", "traffic_quality"]

OWNER = {
    RootCause.landing_page_down: "tech_oncall",
    RootCause.tracking_broken: "tracking_team",
    RootCause.traffic_quality: "affiliate_manager",
    RootCause.offer_cap_reached: "affiliate_manager",
    RootCause.offer_paused: "affiliate_manager",
    RootCause.geo_mismatch: "affiliate_manager",
    RootCause.unknown: "affiliate_manager",
}

ACTIONS = {
    RootCause.landing_page_down: [
        "Page tech on-call to restore the landing page.",
        "Ask the affiliate to pause or redirect traffic to the backup page until it is back.",
        "Re-run triage once the page returns 200 to confirm conversions recover.",
    ],
    RootCause.tracking_broken: [
        "Tracking team: restore the conversion pixel or postback on the landing page.",
        "Check the last landing page deploy for removed tracking code.",
        "Reconcile missed conversions with the advertiser's report once fixed.",
    ],
    RootCause.traffic_quality: [
        "Affiliate manager: review the flagged sub source and decide whether to block it.",
        "Hold payout for the flagged source until the review is done.",
        "Ask the affiliate for the placement behind the flagged sub source.",
    ],
    RootCause.offer_cap_reached: [
        "Tell the affiliate the daily cap is reached so they can shift traffic.",
        "Ask the advertiser for a cap increase if the campaign is profitable.",
    ],
    RootCause.offer_paused: [
        "Confirm with the advertiser why the offer is paused and when it resumes.",
        "Tell the affiliate to stop sending traffic until the offer is active.",
    ],
    RootCause.geo_mismatch: [
        "Tell the affiliate which countries the offer accepts and ask them to fix targeting.",
        "Consider a geo redirect so disallowed traffic goes to a matching offer.",
    ],
    RootCause.unknown: [
        "Affiliate manager: review the evidence manually, automated checks found no cause.",
        "Check with the advertiser for changes on their side (creative, price, checkout).",
    ],
    RootCause.no_anomaly: [],
}


def _ok(evidence: dict[str, Any], tool: str) -> bool:
    return tool in evidence and "error" not in evidence[tool]


def _failed(evidence: dict[str, Any], tool: str) -> bool:
    return tool in evidence and "error" in evidence[tool]


Finding = tuple[RootCause, list[str]]


class _Walk:
    """Result of walking the checks: the first finding wins, later ones become `other_findings`."""

    def __init__(self) -> None:
        self.findings: list[Finding] = []

    def add(self, cause: RootCause, reasons: list[str]) -> None:
        self.findings.append((cause, reasons))

    def result(self, missing: str | None = None) -> tuple[RootCause | None, list[str], str | None, list[str]]:
        if not self.findings:
            return None, [], missing, []
        (cause, reasons), rest = self.findings[0], self.findings[1:]
        others = [f"{c.value}: {' '.join(r)}" for c, r in rest]
        return cause, reasons, missing, others


def _find_cause(evidence: dict[str, Any], s: Settings) -> tuple[RootCause | None, list[str], str | None, list[str]]:
    """Walk the checks in order. Returns (cause, reasons, missing_tool, other_findings).

    Most findings are conclusive and stop the walk. A landing page that is up but slow is not:
    it explains part of the drop, so the walk records it as the primary cause and keeps going
    to catch a second problem (for example the offer cap) as an `other_finding`.
    """
    walk = _Walk()
    if "kpi_snapshot" not in evidence:
        return walk.result("kpi_snapshot")
    if _failed(evidence, "kpi_snapshot"):
        walk.add(RootCause.unknown, ["KPI data could not be loaded, so the drop cannot be confirmed."])
        return walk.result()

    kpi = evidence["kpi_snapshot"]
    if not kpi["significant"]:
        if not kpi["enough_volume"]:
            reason = (f"Only {kpi['current']['clicks_total']} clicks in the window, below the "
                      f"{s.min_clicks_for_signal} needed to trust a change.")
        else:
            reason = (f"{kpi['metric']} moved {kpi['metric_change_pct']}%, inside the normal range "
                      f"(alert threshold {s.drop_threshold_pct}%).")
        walk.add(RootCause.no_anomaly, [reason])
        return walk.result()

    # 1. Landing page
    if "landing_page_check" not in evidence:
        return walk.result("landing_page_check")
    page = evidence["landing_page_check"]
    if _ok(evidence, "landing_page_check") and not page["reachable"]:
        walk.add(RootCause.landing_page_down, [
            f"Landing page returned HTTP {page['status_code']} in {page['latency_ms']} ms.",
            "Clicks are still arriving, so every visitor hits a broken page.",
        ])
        return walk.result()  # a dead page explains everything, stop here
    if _ok(evidence, "landing_page_check") and page["slow"]:
        walk.add(RootCause.landing_page_down, [
            f"Landing page returned HTTP {page['status_code']} but took {page['latency_ms']} ms, "
            f"over the {s.page_latency_limit_ms} ms limit.",
            "Many visitors leave before the page loads.",
        ])  # not conclusive: keep checking for a second cause

    # 2. Offer
    if "offer_status" not in evidence:
        return walk.result("offer_status")
    if _ok(evidence, "offer_status"):
        offer = evidence["offer_status"]
        if offer["status"] != "active":
            walk.add(RootCause.offer_paused, [f"Offer {offer['offer_id']} is {offer['status']} at the advertiser."])
            return walk.result()
        if offer["cap_reached"]:
            walk.add(RootCause.offer_cap_reached, [
                f"Offer {offer['offer_id']} hit its daily cap "
                f"({offer['conversions_today']}/{offer['daily_cap']}); further conversions are rejected.",
            ])
            return walk.result()

    # 3. Tracking
    if "tracking_check" not in evidence:
        return walk.result("tracking_check")
    if _ok(evidence, "landing_page_check") and not page["has_tracking_pixel"]:
        walk.add(RootCause.tracking_broken, ["Landing page is up but the tracking pixel is missing from the page."])
        return walk.result()
    if _ok(evidence, "tracking_check"):
        tracking = evidence["tracking_check"]
        ratio = tracking["postback_ratio"]
        if tracking["pixel_events"] == 0 and tracking["clicks_recorded"] >= s.min_clicks_for_signal:
            walk.add(RootCause.tracking_broken, [f"{tracking['clicks_recorded']} clicks recorded but zero pixel events."])
            return walk.result()
        if ratio is not None and ratio < s.postback_ratio_threshold:
            walk.add(RootCause.tracking_broken, [
                f"Only {tracking['postbacks_received']} postbacks for {tracking['pixel_events']} "
                f"pixel events (ratio {ratio}).",
            ])
            return walk.result()

    # 4. Traffic
    if "traffic_quality" not in evidence:
        return walk.result("traffic_quality")
    if _ok(evidence, "traffic_quality"):
        traffic = evidence["traffic_quality"]
        worst = traffic["worst_source"]
        if worst["bot_score"] >= s.bot_score_threshold and worst["share"] >= s.bad_source_share_threshold:
            walk.add(RootCause.traffic_quality, [
                f"Sub source {worst['sub_id']} sends {round(worst['share'] * 100)}% of clicks "
                f"with bot score {worst['bot_score']}.",
            ])
            return walk.result()
        if traffic["disallowed_geo_share"] >= s.disallowed_geo_share_threshold:
            walk.add(RootCause.geo_mismatch, [
                f"{round(traffic['disallowed_geo_share'] * 100)}% of traffic comes from "
                f"{', '.join(traffic['disallowed_geos'])}, which the offer does not accept.",
            ])
            return walk.result()

    if not walk.findings:
        walk.add(RootCause.unknown, ["The drop is real but landing page, offer, tracking and traffic "
                                     "checks found no cause."])
    return walk.result()


def next_needed(evidence: dict[str, Any], settings: Settings) -> str | None:
    """Which check is still required before a verdict is possible. None means ready."""
    return _find_cause(evidence, settings)[2]


def provisional_cause(evidence: dict[str, Any], settings: Settings) -> RootCause | None:
    return _find_cause(evidence, settings)[0]


def _severity(cause: RootCause, loss: float, s: Settings) -> Severity:
    if cause is RootCause.no_anomaly:
        return Severity.info
    if cause in (RootCause.offer_cap_reached, RootCause.offer_paused):
        # Expected business events, not outages: never page anyone for these.
        return Severity.medium if loss >= s.high_loss_per_hour_usd else Severity.low
    if cause is RootCause.landing_page_down or loss >= s.critical_loss_per_hour_usd:
        return Severity.critical
    if cause in (RootCause.tracking_broken, RootCause.traffic_quality) or loss >= s.high_loss_per_hour_usd:
        return Severity.high
    return Severity.medium


def evaluate(evidence: dict[str, Any], settings: Settings, llm_hypothesis: str | None = None) -> Verdict:
    """Final verdict. Normally called once `next_needed` returns None. If the step budget
    ran out before that, the verdict is `unknown` and goes to a human."""
    cause, reasons, missing, others = _find_cause(evidence, settings)
    if cause is None:
        # Budget ran out before the checks were complete: never guess.
        cause = RootCause.unknown
        reasons = [f"Investigation stopped before '{missing}' could run."]
    elif missing:
        # A non-conclusive finding is kept, but the run is not treated as complete.
        reasons = reasons + [f"Investigation stopped before '{missing}' could run."]

    unavailable = [tool for tool in CHECK_ORDER if _failed(evidence, tool)]
    kpi_ok = _ok(evidence, "kpi_snapshot")
    loss = evidence["kpi_snapshot"]["est_revenue_loss_per_hour_usd"] if kpi_ok else 0.0
    if cause is RootCause.no_anomaly:
        loss = 0.0

    if cause is RootCause.unknown:
        confidence = 0.4
    elif unavailable:
        confidence = 0.7
    else:
        confidence = 0.9
    needs_review = cause is RootCause.unknown or bool(unavailable) or bool(missing)
    if missing and cause is not RootCause.unknown:
        confidence = min(confidence, 0.7)

    if unavailable:
        reasons = reasons + [f"Checks unavailable during this run: {', '.join(unavailable)}."]
    if llm_hypothesis and llm_hypothesis != cause.value:
        confidence = round(max(confidence - 0.2, 0.1), 2)
        needs_review = True
        reasons = reasons + [f"The model's own hypothesis was '{llm_hypothesis}'; rules decided "
                             f"'{cause.value}'. Flagged for review."]

    return Verdict(
        root_cause=cause,
        severity=_severity(cause, loss, settings),
        confidence=confidence,
        needs_human_review=needs_review,
        reasons=reasons,
        recommended_actions=ACTIONS[cause],
        other_findings=others,
        owner_role=OWNER.get(cause),
        est_revenue_loss_per_hour_usd=loss,
        unavailable_checks=unavailable,
    )
