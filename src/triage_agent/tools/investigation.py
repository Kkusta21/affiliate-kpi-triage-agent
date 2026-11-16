"""Read-only investigation tools. Each returns a small, typed-by-convention dict of facts.

Tools return numbers and booleans wherever possible. Free text from outside
(page titles, runbook snippets) goes through `sanitize_untrusted` first.
"""

from __future__ import annotations

import math
import re
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from typing import Any

from ..config import Settings
from ..datasource import DataSource
from ..guardrails import sanitize_untrusted
from ..models import Alert
from ..stats import (
    INCONCLUSIVE,
    NOT_SIGNIFICANT,
    NOT_TESTED,
    SIGNIFICANT,
    TestResult,
    conversion_rate_test,
    epc_bootstrap_test,
    threshold_only,
)


@dataclass
class ToolContext:
    alert: Alert
    campaign: dict[str, Any]
    ds: DataSource
    settings: Settings
    evidence: dict[str, Any] = field(default_factory=dict)
    runbook_query: str | None = None


# ------------------------------------------------------------------ KPI snapshot


def _pct_change(current: float, baseline: float) -> float | None:
    """Percent change, or None when there is no baseline to compare with (zero denominator)."""
    if baseline == 0:
        return None
    return round((current - baseline) / baseline * 100, 1)


def _ts(row: dict[str, Any]) -> datetime:
    parsed = datetime.fromisoformat(str(row["ts"]).replace("Z", "+00:00"))
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)


def _window(rows: list[dict[str, Any]], hours_expected: int) -> dict[str, Any]:
    """Per-hour rates over the buckets that exist, so a missing bucket does not read as zero traffic."""
    hours = len(rows)
    clicks = sum(int(r.get("clicks", 0)) for r in rows)
    conversions = sum(int(r.get("conversions", 0)) for r in rows)
    revenue = float(sum(r.get("revenue_usd", 0.0) for r in rows))
    return {
        "hours_found": hours,
        "hours_expected": hours_expected,
        "clicks_total": clicks,
        "conversions_total": conversions,
        "revenue_total_usd": round(revenue, 2),
        "clicks_per_hour": round(clicks / hours, 1) if hours else 0.0,
        "conversions_per_hour": round(conversions / hours, 2) if hours else 0.0,
        "conversion_rate": round(conversions / clicks, 4) if clicks else 0.0,
        "revenue_per_hour_usd": round(revenue / hours, 2) if hours else 0.0,
        "epc_usd": round(revenue / clicks, 4) if clicks else 0.0,
    }


def _two_proportion_z(c1: int, n1: int, c2: int, n2: int) -> float | None:
    """z score of conversion rate 2 vs rate 1 (negative = current is lower). None without data."""
    if n1 <= 0 or n2 <= 0:
        return None
    pooled = (c1 + c2) / (n1 + n2)
    se = math.sqrt(pooled * (1 - pooled) * (1 / n1 + 1 / n2))
    if se == 0:
        return None
    return round((c2 / n2 - c1 / n1) / se, 2)


def _select_windows(
    rows: list[dict[str, Any]], observed_at: datetime, s: Settings
) -> tuple[list[dict], list[dict], dict[date, list[dict]], bool, str]:
    """Split history into (current, baseline, seasonal rows by day, timestamped?, day type).

    Timestamped rows are selected by time relative to the alert. The seasonal set holds, for
    each earlier day of the same type (weekday or weekend), the rows for the same hours of
    the day as the current window. Rows without timestamps fall back to positional windows.
    """
    if not rows or "ts" not in rows[0]:
        current = rows[-s.current_hours :] if rows else []
        baseline = rows[-(s.current_hours + s.baseline_hours) : -s.current_hours] if len(rows) > s.current_hours else []
        return current, baseline, {}, False, "unknown"

    end = observed_at if observed_at.tzinfo else observed_at.replace(tzinfo=timezone.utc)
    end = end.replace(minute=0, second=0, microsecond=0)  # only complete hour buckets count
    cur_start = end - timedelta(hours=s.current_hours)
    base_start = cur_start - timedelta(hours=s.baseline_hours)
    weekend = cur_start.weekday() >= 5
    hours_of_day = {(cur_start + timedelta(hours=i)).hour for i in range(s.current_hours)}

    current, baseline = [], []
    seasonal: dict[date, list[dict]] = defaultdict(list)
    for row in rows:
        t = _ts(row)
        if cur_start <= t < end:
            current.append(row)
        elif base_start <= t < cur_start:
            baseline.append(row)
        if t.date() < cur_start.date() and t.hour in hours_of_day and (t.weekday() >= 5) == weekend:
            seasonal[t.date()].append(row)
    full_days = {day: day_rows for day, day_rows in seasonal.items() if len(day_rows) == s.current_hours}
    return current, baseline, full_days, True, "weekend_days" if weekend else "weekdays"


def _significance_test(
    metric: str, baseline_rows: list[dict[str, Any]], baseline: dict[str, Any], current: dict[str, Any], s: Settings
) -> TestResult:
    """Pick the test that fits the alert metric. See stats.py for the methods and their limits."""
    if metric == "conversion_rate":
        return conversion_rate_test(
            baseline["conversions_total"],
            baseline["clicks_total"],
            current["conversions_total"],
            current["clicks_total"],
            s.significance_alpha,
        )
    if metric == "epc":
        hours = [
            (int(r.get("clicks", 0)), int(r.get("conversions", 0)), float(r.get("revenue_usd", 0.0))) for r in baseline_rows
        ]
        return epc_bootstrap_test(
            hours,
            current["clicks_total"],
            current["revenue_total_usd"],
            s.significance_alpha,
            s.epc_min_baseline_conversions,
            s.epc_bootstrap_draws,
        )
    return threshold_only(metric, current["revenue_per_hour_usd"] if metric == "revenue" else current["clicks_per_hour"])


def kpi_snapshot(ctx: ToolContext) -> dict[str, Any]:
    """Compare the current window with the rolling baseline and, when history allows, with the
    same hours on earlier days of the same type. Decide whether the drop is a real signal.

    Every number is explainable: per-hour rates, a percent change, a significance test that fits
    the metric (stats.py), and explicit data quality counts. No model is involved.
    """
    s = ctx.settings
    rows = ctx.ds.get_hourly_kpi(ctx.campaign["id"])
    current_rows, baseline_rows, seasonal_days, timestamped, day_type = _select_windows(rows, ctx.alert.observed_at, s)
    baseline = _window(baseline_rows, s.baseline_hours)
    current = _window(current_rows, s.current_hours)
    metric = ctx.alert.metric.value

    data_sufficient = current["hours_found"] >= s.min_current_hours and baseline["hours_found"] >= s.min_baseline_hours
    data_quality = {
        "timestamped": timestamped,
        "history_hours": len(rows),
        "missing_current_hours": s.current_hours - current["hours_found"],
        "missing_baseline_hours": s.baseline_hours - baseline["hours_found"],
    }

    change = {
        "clicks": _pct_change(current["clicks_per_hour"], baseline["clicks_per_hour"]),
        "conversion_rate": _pct_change(current["conversion_rate"], baseline["conversion_rate"]),
        "revenue": _pct_change(current["revenue_per_hour_usd"], baseline["revenue_per_hour_usd"]),
        "epc": _pct_change(current["epc_usd"], baseline["epc_usd"]),
    }
    metric_change = change[metric]

    # Volume: for a clicks alert the current window is by definition small, so judge on the baseline.
    volume = baseline["clicks_per_hour"] * s.current_hours if metric == "clicks" else current["clicks_total"]
    enough_volume = volume >= s.min_clicks_for_signal and baseline["clicks_total"] >= s.min_clicks_for_signal

    # Two separate questions (see stats.py): is the change more than chance (statistical), and is it
    # big enough to matter (the business threshold, a heuristic)? Both must hold.
    z = _two_proportion_z(
        baseline["conversions_total"], baseline["clicks_total"], current["conversions_total"], current["clicks_total"]
    )  # informational
    test = _significance_test(metric, baseline_rows, baseline, current, s)
    statistically_significant = {SIGNIFICANT: True, NOT_SIGNIFICANT: False}.get(test.result)
    if test.result == SIGNIFICANT and test.p_value is not None:
        strength = "strong" if test.p_value < 0.001 else "moderate"
    elif test.result == NOT_TESTED and metric_change is not None and metric_change <= s.drop_threshold_pct:
        # No test exists for this metric: the size of the drop is all we have, so never "strong".
        strength = "moderate" if metric_change <= 2 * s.drop_threshold_pct else "weak"
    else:
        strength = "none"

    # Seasonal comparison: the same hours on earlier days of the same type.
    seasonal: dict[str, Any] = {"available": False, "day_type": day_type, "days_used": len(seasonal_days)}
    seasonal_explained = False
    if len(seasonal_days) >= s.seasonal_min_days:
        season_rows = [row for day_rows in seasonal_days.values() for row in day_rows]
        season = _window(season_rows, len(season_rows))
        season_change = {
            "clicks": _pct_change(current["clicks_per_hour"], season["clicks_per_hour"]),
            "conversion_rate": _pct_change(current["conversion_rate"], season["conversion_rate"]),
            "revenue": _pct_change(current["revenue_per_hour_usd"], season["revenue_per_hour_usd"]),
            "epc": _pct_change(current["epc_usd"], season["epc_usd"]),
        }
        seasonal = {
            "available": True,
            "day_type": day_type,
            "days_used": len(seasonal_days),
            "clicks_per_hour": season["clicks_per_hour"],
            "conversion_rate": season["conversion_rate"],
            "revenue_per_hour_usd": season["revenue_per_hour_usd"],
            "epc_usd": season["epc_usd"],
            "metric_change_pct": season_change[metric],
        }
        rolling_drop = metric_change is not None and metric_change <= s.drop_threshold_pct
        seasonal_metric_change = season_change[metric]
        seasonal_explained = bool(
            rolling_drop and seasonal_metric_change is not None and seasonal_metric_change > s.drop_threshold_pct
        )

    candidate = bool(
        data_sufficient
        and enough_volume
        and metric_change is not None
        and metric_change <= s.drop_threshold_pct
        and not seasonal_explained
    )
    significant = candidate and test.result in (SIGNIFICANT, NOT_TESTED)
    # A big drop on enough clicks where the test cannot be run (for example too few baseline
    # conversions for EPC): neither confirmed nor ruled out, so the agent abstains.
    test_inconclusive = candidate and test.result == INCONCLUSIVE
    if not significant:
        strength = "none"  # the strength of a signal that was not accepted would only mislead
    expected_conversions = baseline["conversions_per_hour"] * current["hours_found"]
    loss = round(max(baseline["revenue_per_hour_usd"] - current["revenue_per_hour_usd"], 0.0), 2)
    return {
        "metric": metric,
        "observed_at": ctx.alert.observed_at.isoformat(),
        "baseline": baseline,
        "current": current,
        "change_pct": change,
        "metric_change_pct": metric_change,
        "conversion_z": z,
        "data_sufficient": data_sufficient,
        "data_quality": data_quality,
        "enough_volume": enough_volume,
        "significance": test.as_dict(),
        "statistically_significant": statistically_significant,
        "signal_strength": strength,
        "significant": significant,
        "test_inconclusive": test_inconclusive,
        "traffic_spike": bool(change["clicks"] is not None and change["clicks"] >= s.spike_threshold_pct),
        "conversions_stopped": bool(current["conversions_total"] == 0 and expected_conversions >= 5),
        "seasonal": seasonal,
        "seasonal_explained": seasonal_explained,
        "est_revenue_loss_per_hour_usd": loss,
    }


# ------------------------------------------------------------------ landing page


def classify_status(status: int | None) -> str:
    """One word per outcome. Only a 2xx counts as a page a visitor can convert on."""
    if status is None:
        return "no_response"
    if 200 <= status < 300:
        return "ok"
    if 300 <= status < 400:
        return "redirect"
    if 400 <= status < 500:
        return "client_error"
    return "server_error"


def landing_page_check(ctx: ToolContext) -> dict[str, Any]:
    """Probe the campaign landing page. The URL comes from the campaign record, never from a model."""
    url = ctx.campaign["landing_page_url"]
    response = ctx.ds.fetch_landing_page(url)
    status = response.get("status_code")
    latency = int(response.get("latency_ms", 0) or 0)
    outcome = classify_status(status)
    title, flagged = sanitize_untrusted(response.get("title"), max_len=80)
    return {
        "url": url,
        "status_code": status,
        "outcome": outcome,
        "reachable": outcome == "ok",
        "latency_ms": latency,
        "slow": latency > ctx.settings.page_latency_limit_ms,
        "has_tracking_pixel": bool(response.get("has_tracking_pixel")),
        "title": title,
        "untrusted_text_flagged": flagged,
    }


# ------------------------------------------------------------------ offer, tracking, traffic


def offer_status(ctx: ToolContext) -> dict[str, Any]:
    offer = ctx.ds.get_offer(ctx.campaign["offer_id"])
    cap = int(offer["daily_cap"])
    today = int(offer["conversions_today"])
    return {
        "offer_id": offer["id"],
        "status": offer["status"],
        "daily_cap": cap,
        "conversions_today": today,
        "cap_used_share": round(today / cap, 2) if cap else None,
        "cap_reached": bool(cap) and today >= cap,
        "allowed_geos": offer["allowed_geos"],
    }


def tracking_check(ctx: ToolContext) -> dict[str, Any]:
    """Do clicks turn into pixel events, and pixel events into postbacks?"""
    stats = ctx.ds.get_tracking_stats(ctx.campaign["id"])
    pixel = int(stats["pixel_events"])
    ratio = round(stats["postbacks_received"] / pixel, 2) if pixel else None
    return {
        "clicks_recorded": stats["clicks_recorded"],
        "pixel_events": pixel,
        "postbacks_received": stats["postbacks_received"],
        "postback_ratio": ratio,
        "postback_errors": stats["postback_errors"],
        "minutes_since_last_postback": stats["minutes_since_last_postback"],
    }


def traffic_quality(ctx: ToolContext) -> dict[str, Any]:
    """Break current traffic down by sub source and country."""
    data = ctx.ds.get_traffic_breakdown(ctx.campaign["id"])
    allowed = set(ctx.ds.get_offer(ctx.campaign["offer_id"])["allowed_geos"])
    total = sum(src["clicks"] for src in data["sources"]) or 1
    sources = [
        {
            "sub_id": src["sub_id"],
            "clicks": src["clicks"],
            "share": round(src["clicks"] / total, 2),
            "bot_score": src["bot_score"],
            "top_geo": src["top_geo"],
        }
        for src in data["sources"]
    ]
    worst = max(sources, key=lambda src: src["bot_score"])
    disallowed = {geo: share for geo, share in data["geo_share"].items() if geo not in allowed}
    return {
        "sources": sources,
        "worst_source": worst,
        "geo_share": data["geo_share"],
        "disallowed_geos": sorted(disallowed),
        "disallowed_geo_share": round(sum(disallowed.values()), 2),
    }


# ------------------------------------------------------------------ runbooks

_WORD_RE = re.compile(r"[a-z0-9]+")


def _tokens(text: str) -> list[str]:
    return _WORD_RE.findall(text.lower())


def runbook_search(ctx: ToolContext) -> dict[str, Any]:
    """Keyword search over the local runbooks (markdown, split by heading)."""
    query = ctx.runbook_query or ctx.alert.metric.value.replace("_", " ")
    query_tokens = set(_tokens(query))
    scored: list[tuple[int, str, str, str]] = []
    for path in sorted(ctx.settings.runbooks_dir.glob("*.md")):
        sections = re.split(r"^##\s+", path.read_text(), flags=re.MULTILINE)
        for section in sections[1:]:
            heading, _, body = section.partition("\n")
            tokens = _tokens(heading + " " + body) + _tokens(path.stem) * 3
            score = sum(tokens.count(tok) for tok in query_tokens)
            if score:
                scored.append((score, path.name, heading.strip(), body.strip()))
    scored.sort(key=lambda item: (-item[0], item[1], item[2]))
    hits = []
    for score, source, heading, body in scored[:2]:
        safe_heading, heading_flagged = sanitize_untrusted(heading, max_len=80)
        snippet, snippet_flagged = sanitize_untrusted(body, max_len=320)
        hits.append(
            {
                "source": source,
                "heading": safe_heading,
                "snippet": snippet,
                "score": score,
                "untrusted_text_flagged": heading_flagged or snippet_flagged,
            }
        )
    return {"query": query, "hits": hits}
