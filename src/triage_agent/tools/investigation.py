"""Read-only investigation tools. Each returns a small, typed-by-convention dict of facts.

Tools return numbers and booleans wherever possible. Free text from outside
(page titles, runbook snippets) goes through `sanitize_untrusted` first.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from ..config import Settings
from ..datasource import DataSource
from ..guardrails import sanitize_untrusted
from ..models import Alert


@dataclass
class ToolContext:
    alert: Alert
    campaign: dict[str, Any]
    ds: DataSource
    settings: Settings
    evidence: dict[str, Any] = field(default_factory=dict)
    runbook_query: str | None = None


def _pct_change(current: float, baseline: float) -> float:
    if baseline == 0:
        return 0.0
    return round((current - baseline) / baseline * 100, 1)


def _window(rows: list[dict[str, Any]]) -> dict[str, float]:
    hours = max(len(rows), 1)
    clicks = sum(r["clicks"] for r in rows)
    conversions = sum(r["conversions"] for r in rows)
    revenue = sum(r["revenue_usd"] for r in rows)
    return {
        "clicks_total": clicks,
        "conversions_total": conversions,
        "clicks_per_hour": round(clicks / hours, 1),
        "conversion_rate": round(conversions / clicks, 4) if clicks else 0.0,
        "revenue_per_hour_usd": round(revenue / hours, 2),
        "epc_usd": round(revenue / clicks, 4) if clicks else 0.0,
    }


def kpi_snapshot(ctx: ToolContext) -> dict[str, Any]:
    """Compare the last hours with the baseline and decide if the drop is a real signal."""
    s = ctx.settings
    rows = ctx.ds.get_hourly_kpi(ctx.campaign["id"])
    baseline = _window(rows[: s.baseline_hours])
    current = _window(rows[s.baseline_hours: s.baseline_hours + s.current_hours])
    change = {
        "clicks": _pct_change(current["clicks_per_hour"], baseline["clicks_per_hour"]),
        "conversion_rate": _pct_change(current["conversion_rate"], baseline["conversion_rate"]),
        "revenue": _pct_change(current["revenue_per_hour_usd"], baseline["revenue_per_hour_usd"]),
        "epc": _pct_change(current["epc_usd"], baseline["epc_usd"]),
    }
    metric = ctx.alert.metric.value
    # For a clicks alert the current window is by definition small, so judge volume on the baseline.
    volume = (baseline["clicks_per_hour"] * s.current_hours if metric == "clicks"
              else current["clicks_total"])
    enough_volume = volume >= s.min_clicks_for_signal
    metric_change = change[metric]
    loss = round(max(baseline["revenue_per_hour_usd"] - current["revenue_per_hour_usd"], 0.0), 2)
    return {
        "metric": metric,
        "baseline": baseline,
        "current": current,
        "change_pct": change,
        "metric_change_pct": metric_change,
        "enough_volume": enough_volume,
        "significant": bool(enough_volume and metric_change <= s.drop_threshold_pct),
        "est_revenue_loss_per_hour_usd": loss,
    }


def landing_page_check(ctx: ToolContext) -> dict[str, Any]:
    """Probe the campaign landing page. The URL comes from the campaign record, never from a model."""
    url = ctx.campaign["landing_page_url"]
    response = ctx.ds.fetch_landing_page(url)
    status = response.get("status_code")
    latency = response.get("latency_ms", 0)
    title, flagged = sanitize_untrusted(response.get("title"), max_len=80)
    return {
        "url": url,
        "status_code": status,
        "reachable": status is not None and status < 500,
        "latency_ms": latency,
        "slow": latency > ctx.settings.page_latency_limit_ms,
        "has_tracking_pixel": bool(response.get("has_tracking_pixel")),
        "title": title,
        "untrusted_text_flagged": flagged,
    }


def offer_status(ctx: ToolContext) -> dict[str, Any]:
    offer = ctx.ds.get_offer(ctx.campaign["offer_id"])
    return {
        "offer_id": offer["id"],
        "status": offer["status"],
        "daily_cap": offer["daily_cap"],
        "conversions_today": offer["conversions_today"],
        "cap_reached": offer["conversions_today"] >= offer["daily_cap"],
        "allowed_geos": offer["allowed_geos"],
    }


def tracking_check(ctx: ToolContext) -> dict[str, Any]:
    """Do clicks turn into pixel events, and pixel events into postbacks?"""
    stats = ctx.ds.get_tracking_stats(ctx.campaign["id"])
    pixel = stats["pixel_events"]
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
        {"sub_id": src["sub_id"], "clicks": src["clicks"], "share": round(src["clicks"] / total, 2),
         "bot_score": src["bot_score"], "top_geo": src["top_geo"]}
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
        hits.append({"source": source, "heading": safe_heading, "snippet": snippet, "score": score,
                     "untrusted_text_flagged": heading_flagged or snippet_flagged})
    return {"query": query, "hits": hits}
