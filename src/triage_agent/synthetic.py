"""Deterministic synthetic KPI history. Used by the demo fixtures and the evaluation dataset.

Nothing here is real campaign data. The generator exists so that every number in the demo
and in the evaluation can be reproduced exactly from a short, readable specification.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any, Iterable, Sequence

DEFAULT_END = datetime(2026, 11, 16, 12, tzinfo=timezone.utc)  # a Monday; the last bucket ends here

# Payout per conversion varies by order size. Cycled per hour when `payout_pattern` is on, so
# EPC has realistic, skewed noise (a few large orders).
PAYOUT_PATTERN = (1.0, 0.6, 1.5, 0.8, 1.2, 0.9, 2.2, 0.7, 1.1, 0.5, 1.3, 0.95)


def hourly_history(
    base_clicks: int,
    base_cr: float,
    payout: float,
    current: Sequence[Sequence[float]] | None = None,
    days: int = 8,
    weekend_factor: float = 1.0,
    drop: Iterable[str] = (),
    end: datetime = DEFAULT_END,
    payout_pattern: bool = False,
) -> list[dict[str, Any]]:
    """`days` days of hourly buckets with a small fixed wobble, ending at `end`.

    - `current` replaces the last buckets, one entry per hour: (clicks, conversions) or
      (clicks, conversions, revenue).
    - `weekend_factor` scales clicks and conversion rate on Saturdays and Sundays.
    - `drop` removes buckets by "YYYY-MM-DDTHH" to simulate gaps in a tracker export.
    - `payout_pattern` makes the payout per conversion vary hour by hour (see PAYOUT_PATTERN).
    """
    rows: list[dict[str, Any]] = []
    total = days * 24
    for i in range(total):
        ts = end - timedelta(hours=total - i)
        position = i % 24
        clicks = base_clicks + ((position * 7) % 11) - 5
        cr = base_cr
        if ts.weekday() >= 5:
            clicks = round(clicks * weekend_factor)
            cr = base_cr * weekend_factor
        conversions = round(clicks * cr)
        hour_payout = payout * (PAYOUT_PATTERN[i % len(PAYOUT_PATTERN)] if payout_pattern else 1.0)
        rows.append(
            {
                "ts": ts.isoformat().replace("+00:00", "Z"),
                "hour": ts.hour,
                "clicks": clicks,
                "conversions": conversions,
                "revenue_usd": round(conversions * hour_payout, 2),
            }
        )
    if current:
        for row, values in zip(rows[-len(current) :], current):
            clicks, conversions = int(values[0]), int(values[1])
            revenue = float(values[2]) if len(values) > 2 else conversions * payout
            row.update({"clicks": clicks, "conversions": conversions, "revenue_usd": round(revenue, 2)})
    dropped = set(drop)
    return [row for row in rows if str(row["ts"])[:13] not in dropped]
