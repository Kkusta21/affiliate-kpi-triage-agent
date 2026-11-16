"""KPI maths on hand-built history: windows, gaps, zero denominators, significance, seasonality."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

from triage_agent import rules
from triage_agent.config import SETTINGS
from triage_agent.datasource import DataSource
from triage_agent.models import Alert, RootCause
from triage_agent.tools import INVESTIGATION_TOOLS, ToolContext

from conftest import alert

END = datetime(2026, 11, 16, 12, tzinfo=timezone.utc)   # a Monday


class History:
    """A data source that serves one hand-built KPI history."""

    def __init__(self, rows):
        self.rows = rows

    def get_hourly_kpi(self, campaign_id):
        return self.rows


def history(hours=24, clicks=100, conversions=4, current=None, drop=(), end=END, weekend_scale=1.0):
    """`hours` hourly buckets ending at `end`. `current` overrides the last buckets,
    `drop` removes buckets by how many hours before `end` they start."""
    rows = []
    for back in range(hours, 0, -1):
        ts = end - timedelta(hours=back)
        scale = weekend_scale if ts.weekday() >= 5 else 1.0
        converted = round(conversions * scale * scale)
        rows.append({"ts": ts.isoformat(), "clicks": round(clicks * scale), "conversions": converted,
                     "revenue_usd": converted * 10.0})
    for row, (c, v) in zip(rows[-len(current or []):] if current else [], current or []):
        row.update(clicks=c, conversions=v, revenue_usd=v * 10.0)
    return [row for back, row in zip(range(hours, 0, -1), rows) if back not in drop]


def snapshot(rows, **alert_kw):
    ctx = ToolContext(alert=Alert.model_validate(alert(**alert_kw)), campaign={"id": "CMP-1"},
                      ds=History(rows), settings=SETTINGS)
    return INVESTIGATION_TOOLS["kpi_snapshot"].fn(ctx)


def test_a_clear_drop_is_significant_with_supporting_statistics():
    kpi = snapshot(history(current=[(100, 0), (100, 0), (100, 1)]))
    assert kpi["significant"] and kpi["data_sufficient"] and kpi["statistically_significant"]
    assert kpi["metric_change_pct"] < -90 and kpi["conversion_z"] < -2.58
    assert kpi["baseline"]["hours_found"] == 21 and kpi["current"]["hours_found"] == 3
    assert kpi["est_revenue_loss_per_hour_usd"] > 30


def test_a_missing_bucket_is_not_counted_as_an_hour_of_zero_traffic():
    full = snapshot(history())
    gap = snapshot(history(drop={2, 10}))     # one current and one baseline bucket missing
    assert gap["data_quality"] == {"timestamped": True, "history_hours": 22,
                                   "missing_current_hours": 1, "missing_baseline_hours": 1}
    assert gap["current"]["clicks_per_hour"] == full["current"]["clicks_per_hour"] == 100.0
    assert gap["change_pct"]["clicks"] == 0.0 and gap["data_sufficient"] and not gap["significant"]


def test_too_few_buckets_is_insufficient_data_not_a_verdict():
    kpi = snapshot(history(current=[(100, 0), (100, 0), (100, 0)], drop={1, 2}))
    assert not kpi["data_sufficient"] and not kpi["significant"]
    verdict = rules.evaluate({"kpi_snapshot": kpi}, SETTINGS)
    assert rules.next_needed({"kpi_snapshot": kpi}, SETTINGS) is None        # no point running more tools
    assert verdict.root_cause is RootCause.unknown and verdict.needs_human_review
    assert "incomplete" in verdict.reasons[0] and verdict.est_revenue_loss_per_hour_usd == 0
    assert verdict.recommended_actions == rules.INSUFFICIENT_DATA_ACTIONS


def test_no_history_at_all_does_not_crash_or_report_healthy():
    kpi = snapshot([])
    assert not kpi["data_sufficient"] and kpi["metric_change_pct"] is None
    assert rules.evaluate({"kpi_snapshot": kpi}, SETTINGS).root_cause is RootCause.unknown


def test_zero_baseline_gives_no_change_value_instead_of_a_fake_zero():
    kpi = snapshot(history(clicks=100, conversions=0, current=[(100, 0)] * 3))
    assert kpi["change_pct"]["conversion_rate"] is None and kpi["change_pct"]["revenue"] is None
    assert kpi["conversion_z"] is None and not kpi["significant"]
    verdict = rules.evaluate({"kpi_snapshot": kpi}, SETTINGS)
    assert verdict.root_cause is RootCause.no_anomaly and "no baseline" in verdict.reasons[0]


def test_zero_current_clicks_does_not_divide_by_zero():
    kpi = snapshot(history(current=[(0, 0)] * 3), metric="clicks")
    assert kpi["current"]["conversion_rate"] == 0.0 and kpi["change_pct"]["clicks"] == -100.0
    assert kpi["significant"]                       # a clicks alert is judged on the baseline volume


def test_a_drop_within_chance_for_the_volume_is_not_significant():
    # 4.0% -> 2.67% is -33%, past the -30% threshold, but on 300 clicks it is not unusual.
    kpi = snapshot(history(current=[(100, 3), (100, 3), (100, 2)]))
    assert kpi["metric_change_pct"] <= -30 and kpi["enough_volume"]
    assert kpi["statistically_significant"] is False and not kpi["significant"]
    verdict = rules.evaluate({"kpi_snapshot": kpi}, SETTINGS)
    assert verdict.root_cause is RootCause.no_anomaly and "within chance" in verdict.reasons[0]


def test_traffic_spike_and_stopped_conversions_are_flagged():
    spike = snapshot(history(current=[(350, 4), (360, 4), (340, 4)]))
    assert spike["traffic_spike"] and spike["change_pct"]["clicks"] >= 100
    stopped = snapshot(history(current=[(100, 0)] * 3))
    assert stopped["conversions_stopped"] and not snapshot(history())["conversions_stopped"]


def test_alert_time_inside_an_hour_uses_the_same_complete_buckets():
    rows = history(current=[(100, 0), (100, 0), (100, 1)])
    on_the_hour = snapshot(rows, observed_at="2026-11-16T12:00:00Z")
    inside = snapshot(rows, observed_at="2026-11-16T12:25:00Z")
    assert inside["current"] == on_the_hour["current"] and inside["baseline"] == on_the_hour["baseline"]


def test_rows_without_timestamps_fall_back_to_positional_windows():
    rows = [{"clicks": 100, "conversions": 4, "revenue_usd": 40.0} for _ in range(21)] + \
           [{"clicks": 100, "conversions": 0, "revenue_usd": 0.0} for _ in range(3)]
    kpi = snapshot(rows)
    assert not kpi["data_quality"]["timestamped"] and not kpi["seasonal"]["available"]
    assert kpi["significant"] and kpi["baseline"]["hours_found"] == 21


def test_seasonal_baseline_needs_enough_earlier_days_of_the_same_type():
    assert not snapshot(history(hours=24))["seasonal"]["available"]            # one day of history
    season = snapshot(history(hours=24 * 8))["seasonal"]                        # Monday: five earlier weekdays
    assert season["available"] and season["day_type"] == "weekdays" and season["days_used"] == 5


def test_weekend_level_is_compared_with_earlier_weekends_not_with_friday():
    saturday = datetime(2026, 11, 14, 12, tzinfo=timezone.utc)
    rows = history(hours=24 * 22, end=saturday, weekend_scale=0.5)
    kpi = snapshot(rows, metric="revenue", observed_at="2026-11-14T12:00:00Z")
    assert kpi["metric_change_pct"] <= -30                      # looks like a drop against Friday
    assert kpi["seasonal"]["day_type"] == "weekend_days" and kpi["seasonal"]["metric_change_pct"] == 0.0
    assert kpi["seasonal_explained"] and not kpi["significant"]
    verdict = rules.evaluate({"kpi_snapshot": kpi}, SETTINGS)
    assert verdict.root_cause is RootCause.no_anomaly and "normal pattern" in verdict.reasons[0]


def test_a_real_drop_on_a_weekend_is_not_explained_away():
    saturday = datetime(2026, 11, 14, 12, tzinfo=timezone.utc)
    rows = history(hours=24 * 22, end=saturday, weekend_scale=0.5, current=[(50, 0)] * 3)
    kpi = snapshot(rows, metric="revenue", observed_at="2026-11-14T12:00:00Z")
    assert kpi["seasonal"]["metric_change_pct"] == -100.0
    assert not kpi["seasonal_explained"] and kpi["significant"]


def test_fixture_history_supports_the_weekend_case_end_to_end(triage):
    result = triage(alert("CMP-110", metric="revenue", observed_at="2026-11-14T12:00:00Z"))
    assert result.root_cause.value == "no_anomaly" and [s.tool for s in result.steps] == ["kpi_snapshot"]
    assert result.incident.action == "none" and result.kpi["seasonal"]["days_used"] == 4
    # the same campaign on a normal Monday shows no drop at all
    monday = triage(alert("CMP-110", metric="revenue"))
    assert monday.root_cause.value == "no_anomaly" and not monday.kpi["seasonal_explained"]


def test_default_data_source_serves_timestamped_history():
    rows = DataSource().get_hourly_kpi("CMP-101")
    assert len(rows) == 8 * 24 and rows[-1]["ts"] == "2026-11-16T11:00:00Z"
