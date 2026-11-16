"""Deterministic diagnosis.

The model plans the investigation and writes prose. The decision itself is made here,
by plain rules over the collected evidence, so the same evidence always gives the same
verdict and every verdict can be explained line by line.

The checks run in a fixed priority order (cheapest and most damaging first):
    KPI signal -> landing page -> offer -> tracking -> traffic

Each check has its own matcher that returns graded findings:
    confirmed     a threshold is clearly met on healthy evidence
    suspected     within `suspect_margin` of a threshold: reported, never the root cause
    inconclusive  the check could not run, so its causes cannot be ruled out

The first CONFIRMED finding in priority order is the root cause and ends the required
investigation once the drop is fully explained (see `_find_cause`). Every other finding
on evidence that was ALREADY gathered is reported as well. No tool is run after the drop
is fully explained just to look for a second cause. The model never adds a finding.
"""

from __future__ import annotations

from typing import Any, Callable, Literal

from .config import Settings
from .models import Finding, FindingStatus, RootCause, Severity, Verdict

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

INSUFFICIENT_DATA_ACTIONS = [
    "Check the tracker export for missing hours or missing conversion data.",
    "Re-run triage once more KPI data for the alert window is available.",
]

# What a failed check leaves unresolved.
UNRESOLVED_BY_TOOL = {
    "landing_page_check": [RootCause.landing_page_down],
    "offer_status": [RootCause.offer_paused, RootCause.offer_cap_reached],
    "tracking_check": [RootCause.tracking_broken],
    "traffic_quality": [RootCause.traffic_quality, RootCause.geo_mismatch],
}

CONFIRMED, SUSPECTED, INCONCLUSIVE = FindingStatus.confirmed, FindingStatus.suspected, FindingStatus.inconclusive
Matcher = Callable[[dict[str, Any], Settings, float], list[Finding]]


def _ok(evidence: dict[str, Any], tool: str) -> bool:
    return tool in evidence and "error" not in evidence[tool]


def _failed(evidence: dict[str, Any], tool: str) -> bool:
    return tool in evidence and "error" in evidence[tool]


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


Strength = Literal["strong", "moderate", "weak"]


def _finding(
    cause: RootCause,
    status: FindingStatus,
    strength: Strength,
    explanation: str,
    loss: float,
    s: Settings,
    /,
    **metrics: Any,
) -> Finding:
    severity = _severity(cause, loss, s) if status is CONFIRMED else Severity.low
    actions = ACTIONS[cause] if status is CONFIRMED else []
    return Finding(
        cause=cause,
        status=status,
        severity=severity,
        evidence_strength=strength,
        explanation=explanation,
        supporting_metrics=metrics,
        recommended_actions=actions,
    )


# ---------------------------------------------------------------- matchers
# Each matcher looks only at evidence that is present and healthy, and returns every
# finding it supports. A matcher never asks for more evidence.


def _page_findings(evidence: dict[str, Any], s: Settings, loss: float) -> list[Finding]:
    if not _ok(evidence, "landing_page_check"):
        return []
    page = evidence["landing_page_check"]
    metrics = {"status_code": page["status_code"], "outcome": page.get("outcome"), "latency_ms": page["latency_ms"]}
    if not page["reachable"]:
        outcome = page.get("outcome", "error").replace("_", " ")
        return [
            _finding(
                RootCause.landing_page_down,
                CONFIRMED,
                "strong",
                f"Landing page returned HTTP {page['status_code']} ({outcome}) in {page['latency_ms']} ms. "
                "Clicks are still arriving, so every visitor hits a broken page.",
                loss,
                s,
                **metrics,
            )
        ]
    if page["slow"]:
        return [
            _finding(
                RootCause.landing_page_down,
                CONFIRMED,
                "moderate",
                f"Landing page returned HTTP {page['status_code']} but took {page['latency_ms']} ms, "
                f"over the {s.page_latency_limit_ms} ms limit. Many visitors leave before it loads.",
                loss,
                s,
                **metrics,
            )
        ]
    if page["latency_ms"] >= s.page_latency_limit_ms * s.suspect_margin:
        return [
            _finding(
                RootCause.landing_page_down,
                SUSPECTED,
                "weak",
                f"Landing page loads in {page['latency_ms']} ms, close to the {s.page_latency_limit_ms} ms limit.",
                loss,
                s,
                **metrics,
            )
        ]
    return []


def _offer_findings(evidence: dict[str, Any], s: Settings, loss: float) -> list[Finding]:
    if not _ok(evidence, "offer_status"):
        return []
    offer = evidence["offer_status"]
    metrics = {
        "offer_id": offer["offer_id"],
        "status": offer["status"],
        "daily_cap": offer["daily_cap"],
        "conversions_today": offer["conversions_today"],
    }
    findings: list[Finding] = []
    if offer["status"] != "active":
        findings.append(
            _finding(
                RootCause.offer_paused,
                CONFIRMED,
                "strong",
                f"Offer {offer['offer_id']} is {offer['status']} at the advertiser.",
                loss,
                s,
                **metrics,
            )
        )
    if offer["cap_reached"]:
        findings.append(
            _finding(
                RootCause.offer_cap_reached,
                CONFIRMED,
                "strong",
                f"Offer {offer['offer_id']} hit its daily cap "
                f"({offer['conversions_today']}/{offer['daily_cap']}); further conversions are rejected.",
                loss,
                s,
                **metrics,
            )
        )
    elif offer.get("cap_used_share") is not None and offer["cap_used_share"] >= s.suspect_margin:
        findings.append(
            _finding(
                RootCause.offer_cap_reached,
                SUSPECTED,
                "weak",
                f"Offer {offer['offer_id']} has used {round(offer['cap_used_share'] * 100)}% of its "
                f"daily cap ({offer['conversions_today']}/{offer['daily_cap']}).",
                loss,
                s,
                **metrics,
            )
        )
    return findings


def _tracking_findings(evidence: dict[str, Any], s: Settings, loss: float) -> list[Finding]:
    # A missing pixel only means something when the page itself loads.
    if _ok(evidence, "landing_page_check"):
        page = evidence["landing_page_check"]
        if page["reachable"] and not page["has_tracking_pixel"]:
            return [
                _finding(
                    RootCause.tracking_broken,
                    CONFIRMED,
                    "strong",
                    "Landing page is up but the tracking pixel is missing from the page.",
                    loss,
                    s,
                    has_tracking_pixel=False,
                    status_code=page["status_code"],
                )
            ]
    if not _ok(evidence, "tracking_check"):
        return []
    tracking = evidence["tracking_check"]
    ratio = tracking["postback_ratio"]
    metrics = {
        "clicks_recorded": tracking["clicks_recorded"],
        "pixel_events": tracking["pixel_events"],
        "postbacks_received": tracking["postbacks_received"],
        "postback_ratio": ratio,
    }
    if tracking["pixel_events"] == 0 and tracking["clicks_recorded"] >= s.min_clicks_for_signal:
        return [
            _finding(
                RootCause.tracking_broken,
                CONFIRMED,
                "strong",
                f"{tracking['clicks_recorded']} clicks recorded but zero pixel events.",
                loss,
                s,
                **metrics,
            )
        ]
    if ratio is not None and ratio < s.postback_ratio_threshold:
        return [
            _finding(
                RootCause.tracking_broken,
                CONFIRMED,
                "moderate",
                f"Only {tracking['postbacks_received']} postbacks for {tracking['pixel_events']} "
                f"pixel events (ratio {ratio}).",
                loss,
                s,
                **metrics,
            )
        ]
    if ratio is not None and ratio < s.postback_ratio_threshold / s.suspect_margin:
        return [
            _finding(
                RootCause.tracking_broken,
                SUSPECTED,
                "weak",
                f"Postback ratio {ratio} is close to the {s.postback_ratio_threshold} limit.",
                loss,
                s,
                **metrics,
            )
        ]
    return []


def _traffic_findings(evidence: dict[str, Any], s: Settings, loss: float) -> list[Finding]:
    if not _ok(evidence, "traffic_quality"):
        return []
    traffic = evidence["traffic_quality"]
    worst = traffic["worst_source"]
    kpi = evidence.get("kpi_snapshot", {})
    findings: list[Finding] = []
    source_metrics = {"sub_id": worst["sub_id"], "share": worst["share"], "bot_score": worst["bot_score"]}
    if worst["bot_score"] >= s.bot_score_threshold and worst["share"] >= s.bad_source_share_threshold:
        findings.append(
            _finding(
                RootCause.traffic_quality,
                CONFIRMED,
                "strong",
                f"Sub source {worst['sub_id']} sends {round(worst['share'] * 100)}% of clicks "
                f"with bot score {worst['bot_score']}.",
                loss,
                s,
                **source_metrics,
            )
        )
    elif (
        worst["bot_score"] >= s.bot_score_threshold * s.suspect_margin
        and worst["share"] >= s.bad_source_share_threshold * s.suspect_margin
    ):
        findings.append(
            _finding(
                RootCause.traffic_quality,
                SUSPECTED,
                "weak",
                f"Sub source {worst['sub_id']} has bot score {worst['bot_score']} on "
                f"{round(worst['share'] * 100)}% of clicks, close to the limits.",
                loss,
                s,
                **source_metrics,
            )
        )
    elif (
        kpi.get("traffic_spike")
        and (kpi.get("change_pct") or {}).get("conversion_rate", 0) is not None
        and (kpi.get("change_pct") or {}).get("conversion_rate", 0) <= s.drop_threshold_pct
    ):
        findings.append(
            _finding(
                RootCause.traffic_quality,
                SUSPECTED,
                "weak",
                f"Clicks rose {kpi['change_pct']['clicks']}% while the conversion rate fell "
                f"{kpi['change_pct']['conversion_rate']}%, but no single sub source is flagged.",
                loss,
                s,
                clicks_change_pct=kpi["change_pct"]["clicks"],
                **source_metrics,
            )
        )
    geo_share = traffic["disallowed_geo_share"]
    geo_metrics = {"disallowed_geo_share": geo_share, "disallowed_geos": traffic["disallowed_geos"]}
    if geo_share >= s.disallowed_geo_share_threshold:
        findings.append(
            _finding(
                RootCause.geo_mismatch,
                CONFIRMED,
                "strong",
                f"{round(geo_share * 100)}% of traffic comes from "
                f"{', '.join(traffic['disallowed_geos'])}, which the offer does not accept.",
                loss,
                s,
                **geo_metrics,
            )
        )
    elif geo_share >= s.disallowed_geo_share_threshold * s.suspect_margin:
        findings.append(
            _finding(
                RootCause.geo_mismatch,
                SUSPECTED,
                "weak",
                f"{round(geo_share * 100)}% of traffic comes from disallowed countries, close "
                f"to the {round(s.disallowed_geo_share_threshold * 100)}% limit.",
                loss,
                s,
                **geo_metrics,
            )
        )
    return findings


# (tool that must have run before this stage can be passed, matcher) in priority order
STAGES: list[tuple[str, Matcher]] = [
    ("landing_page_check", _page_findings),
    ("offer_status", _offer_findings),
    ("tracking_check", _tracking_findings),
    ("traffic_quality", _traffic_findings),
]


def _loss(evidence: dict[str, Any]) -> float:
    kpi = evidence.get("kpi_snapshot", {})
    return float(kpi.get("est_revenue_loss_per_hour_usd", 0.0)) if "error" not in kpi else 0.0


def all_findings(evidence: dict[str, Any], s: Settings) -> list[Finding]:
    """Every finding the gathered evidence supports, in priority order, at most one per cause.
    Confirmed and suspected findings come from the matchers; failed checks add inconclusive ones."""
    loss = _loss(evidence)
    findings: list[Finding] = []
    seen: set[RootCause] = set()
    for tool, matcher in STAGES:
        candidates = matcher(evidence, s, loss) if _ok(evidence, tool) or tool == "tracking_check" else []
        if _failed(evidence, tool):
            candidates = candidates + [
                _finding(
                    cause,
                    INCONCLUSIVE,
                    "weak",
                    f"The {tool} check was unavailable, so {cause.value.replace('_', ' ')} cannot be ruled out.",
                    loss,
                    s,
                    tool=tool,
                )
                for cause in UNRESOLVED_BY_TOOL[tool]
            ]
        for finding in candidates:
            if finding.cause not in seen:
                seen.add(finding.cause)
                findings.append(finding)
    return findings


def _kpi_gate(kpi: dict[str, Any], s: Settings) -> tuple[RootCause, str] | None:
    """Verdicts that follow from the KPI snapshot alone. None means: keep investigating."""
    if not kpi.get("data_sufficient", True):
        q = kpi.get("data_quality", {})
        return RootCause.unknown, (
            f"KPI history is incomplete ({kpi['current']['hours_found']} of "
            f"{s.current_hours} current hours and {kpi['baseline']['hours_found']} of "
            f"{s.baseline_hours} baseline hours found, {q.get('history_hours', 0)} hours of "
            "history), so the drop can neither be confirmed nor ruled out."
        )
    if kpi.get("seasonal_explained"):
        season = kpi["seasonal"]
        return RootCause.no_anomaly, (
            f"{kpi['metric']} is {kpi['metric_change_pct']}% vs the last "
            f"{s.baseline_hours} hours, but {season['metric_change_pct']}% vs the same "
            f"hours on the previous {season['days_used']} "
            f"{season['day_type'].replace('_', ' ')}. "
            "This looks like the normal pattern for this time, not a drop."
        )
    test = kpi.get("significance") or {}
    if kpi.get("test_inconclusive"):
        return RootCause.unknown, (
            f"{kpi['metric']} moved {kpi['metric_change_pct']}%, but the statistical test is "
            f"inconclusive ({test.get('detail', 'not enough data')}), so the drop can neither "
            "be confirmed nor ruled out."
        )
    if not kpi["significant"]:
        if not kpi["enough_volume"]:
            return RootCause.no_anomaly, (
                f"Only {kpi['current']['clicks_total']} clicks in the window, below the "
                f"{s.min_clicks_for_signal} needed to trust a change."
            )
        if kpi.get("metric_change_pct") is None:
            return RootCause.no_anomaly, f"{kpi['metric']} had no baseline to compare with (baseline is zero)."
        if kpi.get("statistically_significant") is False:
            return RootCause.no_anomaly, (
                f"{kpi['metric']} moved {kpi['metric_change_pct']}%, but the change is within "
                f"chance for this volume ({test.get('method')}, one-sided p = "
                f"{test.get('p_value')}, alpha {test.get('alpha')})."
            )
        return RootCause.no_anomaly, (
            f"{kpi['metric']} moved {kpi['metric_change_pct']}%, inside the normal range "
            f"(alert threshold {s.drop_threshold_pct}%)."
        )
    return None


def _find_cause(evidence: dict[str, Any], s: Settings) -> tuple[RootCause | None, list[str], str | None, str]:
    """Walk the stages in order. Returns (cause, reasons, missing_tool, stop_reason).

    A stage can only be passed once its tool has run, which is what makes the coverage check
    in the agent possible. The walk stops at the first CONFIRMED finding that fully explains
    the drop (evidence strength "strong": dead page, paused or capped offer, missing pixel,
    bot source, wrong countries). A confirmed finding that explains it only partly (strength
    "moderate": a slow page, a weak postback ratio) becomes the root cause, but the walk
    goes on, because the rest of the drop still needs an explanation.
    """
    if "kpi_snapshot" not in evidence:
        return None, [], "kpi_snapshot", "kpi_snapshot has not run"
    if _failed(evidence, "kpi_snapshot"):
        return (
            RootCause.unknown,
            ["KPI data could not be loaded, so the drop cannot be confirmed."],
            None,
            "kpi_snapshot unavailable",
        )
    gate = _kpi_gate(evidence["kpi_snapshot"], s)
    if gate:
        cause, reason = gate
        return cause, [reason], None, "decided from the KPI snapshot alone"

    loss = _loss(evidence)
    partial: Finding | None = None  # first partly-explaining finding, kept as the root cause
    partial_tool = ""
    for tool, matcher in STAGES:
        if tool not in evidence:
            if partial:
                return partial.cause, [partial.explanation], tool, f"partly explained by {partial_tool}"
            return None, [], tool, f"waiting for {tool}"
        confirmed = (
            [f for f in matcher(evidence, s, loss) if f.status is CONFIRMED]
            if _ok(evidence, tool) or tool == "tracking_check"
            else []
        )
        if not confirmed:
            continue
        if partial is None and confirmed[0].evidence_strength == "moderate":
            partial, partial_tool = confirmed[0], tool
            continue
        primary = partial or confirmed[0]
        return primary.cause, [primary.explanation], None, f"confirmed by {partial_tool or tool}"
    if partial:
        return partial.cause, [partial.explanation], None, f"confirmed by {partial_tool}, no further cause found"

    return (
        RootCause.unknown,
        ["The drop is real but landing page, offer, tracking and traffic checks found no confirmed cause."],
        None,
        "all checks ran, none confirmed a cause",
    )


def next_needed(evidence: dict[str, Any], settings: Settings) -> str | None:
    """Which check is still required before a verdict is possible. None means ready."""
    return _find_cause(evidence, settings)[2]


def provisional_cause(evidence: dict[str, Any], settings: Settings) -> RootCause | None:
    return _find_cause(evidence, settings)[0]


def evaluate(evidence: dict[str, Any], settings: Settings, llm_hypothesis: str | None = None) -> Verdict:
    """Final verdict. Normally called once `next_needed` returns None. If the step budget
    ran out before that, the verdict is `unknown` and goes to a human."""
    s = settings
    cause, reasons, missing, stop_reason = _find_cause(evidence, s)
    if cause is None:
        # Budget ran out before the checks were complete: never guess.
        cause = RootCause.unknown
        reasons = [f"Investigation stopped before '{missing}' could run."]
        stop_reason = "step budget reached before the required checks were complete"
    elif missing:
        # A partly-explaining finding is kept, but the run is not treated as complete.
        reasons = reasons + [f"Investigation stopped before '{missing}' could run."]
        stop_reason = "step budget reached before the required checks were complete"

    kpi_ok = _ok(evidence, "kpi_snapshot")
    kpi = evidence["kpi_snapshot"] if kpi_ok else {}
    # No loss figure when there is no anomaly, or when the KPI data is too thin to trust one.
    kpi_untrusted = not kpi.get("data_sufficient", True) or bool(kpi.get("test_inconclusive"))
    loss = 0.0 if cause is RootCause.no_anomaly or kpi_untrusted else _loss(evidence)
    unavailable = [tool for tool in CHECK_ORDER if _failed(evidence, tool)]

    kpi_trusted = kpi_ok and kpi.get("data_sufficient", True) and not kpi.get("test_inconclusive")
    findings = all_findings(evidence, s) if cause is not RootCause.no_anomaly and kpi_trusted else []
    primary = next((f for f in findings if f.cause is cause and f.status is CONFIRMED), None)
    if primary:
        primary.primary = True
        findings = [primary] + [f for f in findings if f is not primary]
    others = [
        f"{f.cause.value}{'' if f.status is CONFIRMED else f' ({f.status.value})'}: {f.explanation}"
        for f in findings
        if not f.primary
    ]

    if cause is RootCause.unknown:
        confidence = 0.4
    elif unavailable or missing:
        confidence = 0.7
    else:
        confidence = 0.9
    needs_review = cause is RootCause.unknown or bool(unavailable) or bool(missing)

    if cause is not RootCause.no_anomaly and kpi.get("signal_strength") == "weak":
        reasons = reasons + ["The KPI drop itself is only weakly significant for this volume."]
        confidence = round(min(confidence, 0.7), 2)
        needs_review = True
    if unavailable:
        reasons = reasons + [f"Checks unavailable during this run: {', '.join(unavailable)}."]
    if cause is RootCause.unknown and any(f.status is SUSPECTED for f in findings):
        reasons = reasons + [
            "Suspected but unconfirmed: "
            + "; ".join(f"{f.cause.value} ({f.explanation})" for f in findings if f.status is SUSPECTED)
        ]
    if llm_hypothesis and llm_hypothesis != cause.value:
        confidence = round(max(confidence - 0.2, 0.1), 2)
        needs_review = True
        reasons = reasons + [
            f"The model's own hypothesis was '{llm_hypothesis}'; rules decided '{cause.value}'. Flagged for review."
        ]

    return Verdict(
        root_cause=cause,
        severity=_severity(cause, loss, s),
        confidence=confidence,
        needs_human_review=needs_review,
        reasons=reasons,
        recommended_actions=INSUFFICIENT_DATA_ACTIONS if kpi_untrusted else ACTIONS[cause],
        findings=findings,
        other_findings=others,
        stop_reason=stop_reason,
        owner_role=OWNER.get(cause),
        est_revenue_loss_per_hour_usd=loss,
        unavailable_checks=unavailable,
    )
