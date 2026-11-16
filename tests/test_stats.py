"""The significance tests: method choice, exactness, and the EPC cases a binomial test gets wrong."""

from __future__ import annotations

import math
import random
from datetime import timedelta

import pytest

from triage_agent import rules
from triage_agent.config import SETTINGS
from triage_agent.models import RootCause
from triage_agent.stats import (
    INCONCLUSIVE,
    NOT_SIGNIFICANT,
    SIGNIFICANT,
    _binomial,
    conversion_rate_test,
    epc_bootstrap_test,
    fisher_lower_tail,
    normal_cdf,
)

from test_kpi import END, snapshot

PAYOUTS = [20.0, 12.0, 35.0, 18.0, 25.0, 15.0, 60.0]  # skewed: a few large orders


def epc_history(current, clicks=400, conversions=16, hours=24):
    """Hourly rows with a skewed, varying payout per conversion. `current` gives
    (clicks, conversions, payout per conversion) for the last hours."""
    rows = []
    for back in range(hours, 0, -1):
        payout = PAYOUTS[back % len(PAYOUTS)]
        rows.append(
            {
                "ts": (END - timedelta(hours=back)).isoformat(),
                "clicks": clicks,
                "conversions": conversions,
                "revenue_usd": conversions * payout,
            }
        )
    for row, (c, v, payout) in zip(rows[-len(current) :], current):
        row.update(clicks=c, conversions=v, revenue_usd=v * payout)
    return rows


BASE_PAYOUT = sum(PAYOUTS) / len(PAYOUTS)  # about 26.4


# ---------------------------------------------------------------- method choice and exactness


def test_z_test_matches_a_hand_calculation():
    result = conversion_rate_test(base_conv=84, base_clicks=2100, cur_conv=8, cur_clicks=300, alpha=0.01)
    pooled = 92 / 2400
    z = (8 / 300 - 84 / 2100) / math.sqrt(pooled * (1 - pooled) * (1 / 2100 + 1 / 300))
    assert result.method == "two_proportion_z" and result.statistic == round(z, 2)
    assert result.p_value == pytest.approx(normal_cdf(z), abs=1e-6) and result.result == NOT_SIGNIFICANT


def test_small_expected_counts_switch_to_the_exact_test():
    result = conversion_rate_test(base_conv=6, base_clicks=150, cur_conv=0, cur_clicks=60, alpha=0.01)
    assert result.method == "fisher_exact" and "expected counts below 5" in result.detail
    assert result.p_value == pytest.approx(fisher_lower_tail(0, 60, 6, 150), abs=1e-6)


def test_fisher_lower_tail_matches_the_hypergeometric_formula():
    # 0 of 10 converted now vs 10 of 10 before: P = 1 / C(20, 10)
    assert fisher_lower_tail(0, 10, 10, 10) == pytest.approx(1 / math.comb(20, 10))
    assert fisher_lower_tail(10, 10, 0, 10) == pytest.approx(1.0)


def test_no_conversions_anywhere_is_inconclusive_not_significant():
    result = conversion_rate_test(0, 2000, 0, 300, alpha=0.01)
    assert result.result == INCONCLUSIVE and result.p_value is None


def test_binomial_sampler_is_unbiased_and_handles_edges():
    rng = random.Random(1)
    draws = [_binomial(rng, 1000, 0.04) for _ in range(3000)]
    assert sum(draws) / len(draws) == pytest.approx(40, abs=1.0)
    assert _binomial(rng, 50, 0.0) == 0 and _binomial(rng, 50, 1.0) == 50 and _binomial(rng, 0, 0.5) == 0


def test_bootstrap_is_deterministic():
    hours = [(400, 16, 16 * p) for p in PAYOUTS * 3]
    first = epc_bootstrap_test(hours, 1200, 1200 * 0.6, alpha=0.01)
    assert first == epc_bootstrap_test(hours, 1200, 1200 * 0.6, alpha=0.01)


# ---------------------------------------------------------------- the EPC regressions


def test_unchanged_conversion_rate_with_a_payout_cut_is_a_significant_epc_drop():
    kpi = snapshot(epc_history(current=[(400, 16, 12.0)] * 3), metric="epc")
    assert kpi["change_pct"]["conversion_rate"] == 0.0 and kpi["metric_change_pct"] <= -50
    assert kpi["significance"]["method"] == "parametric_bootstrap" and kpi["significance"]["result"] == SIGNIFICANT
    assert kpi["significant"] and kpi["significance"]["observed"] < kpi["significance"]["expected_low"]
    # The old method, a binomial test on conversions, sees nothing at all here.
    cr = conversion_rate_test(
        kpi["baseline"]["conversions_total"],
        kpi["baseline"]["clicks_total"],
        kpi["current"]["conversions_total"],
        kpi["current"]["clicks_total"],
        0.01,
    )
    assert cr.result == NOT_SIGNIFICANT


def test_falling_conversion_rate_with_stable_epc_is_not_an_epc_incident():
    # Conversion rate halves, payouts double (the affiliate moved to a premium plan): EPC holds.
    kpi = snapshot(epc_history(current=[(400, 8, 2 * BASE_PAYOUT)] * 3), metric="epc")
    assert kpi["change_pct"]["conversion_rate"] == -50.0
    assert kpi["significance"]["result"] == NOT_SIGNIFICANT and not kpi["significant"]
    assert rules.evaluate({"kpi_snapshot": kpi}, SETTINGS).root_cause is RootCause.no_anomaly
    # The same data on a conversion-rate alert IS a drop. The test follows the alert metric.
    assert snapshot(epc_history(current=[(400, 8, 2 * BASE_PAYOUT)] * 3), metric="conversion_rate")["significant"]


def test_epc_dip_past_the_threshold_but_within_payout_noise_is_not_an_incident():
    # -32% EPC crosses the -30% business threshold, but with payouts this skewed three hours
    # of lower order values happen by chance. A threshold-only check would raise an incident.
    kpi = snapshot(epc_history(current=[(400, 16, 18.0)] * 3), metric="epc")
    assert kpi["metric_change_pct"] <= -30
    assert kpi["significance"]["result"] == NOT_SIGNIFICANT and 0.01 < kpi["significance"]["p_value"] < 0.1
    assert kpi["significance"]["expected_low"] <= kpi["significance"]["observed"] <= kpi["significance"]["expected_high"]
    assert not kpi["significant"]
    assert rules.evaluate({"kpi_snapshot": kpi}, SETTINGS).root_cause is RootCause.no_anomaly


def test_small_epc_sample_is_inconclusive_and_the_agent_abstains():
    # 2 conversions a day in the baseline: far too few to know the payout distribution.
    rows = epc_history(current=[(150, 0, 0.0)] * 3, clicks=150, conversions=0)
    for i, row in enumerate(rows[:-3]):
        if i % 12 == 0:
            row.update(conversions=1, revenue_usd=40.0)
    kpi = snapshot(rows, metric="epc")
    assert kpi["significance"]["result"] == INCONCLUSIVE and "baseline conversions" in kpi["significance"]["detail"]
    assert kpi["test_inconclusive"] and not kpi["significant"]
    verdict = rules.evaluate({"kpi_snapshot": kpi}, SETTINGS)
    assert verdict.root_cause is RootCause.unknown and verdict.needs_human_review
    assert "inconclusive" in verdict.reasons[0] and verdict.est_revenue_loss_per_hour_usd == 0
    assert rules.next_needed({"kpi_snapshot": kpi}, SETTINGS) is None


def test_current_revenue_of_zero_is_a_significant_epc_drop():
    kpi = snapshot(epc_history(current=[(400, 0, 0.0)] * 3), metric="epc")
    assert kpi["current"]["revenue_total_usd"] == 0 and kpi["metric_change_pct"] == -100.0
    assert kpi["significance"]["result"] == SIGNIFICANT and kpi["significant"]


def test_baseline_revenue_of_zero_has_nothing_to_fall_from():
    rows = epc_history(current=[(400, 16, 0.0)] * 3)
    for row in rows:
        row["revenue_usd"] = 0.0
    kpi = snapshot(rows, metric="epc")
    assert kpi["metric_change_pct"] is None and not kpi["significant"] and not kpi["test_inconclusive"]
    assert rules.evaluate({"kpi_snapshot": kpi}, SETTINGS).root_cause is RootCause.no_anomaly


def test_epc_test_uses_only_the_hours_that_exist():
    full = snapshot(epc_history(current=[(400, 16, 12.0)] * 3), metric="epc")
    rows = epc_history(current=[(400, 16, 12.0)] * 3)
    gapped = snapshot([row for i, row in enumerate(rows) if i not in (2, 5, 9)], metric="epc")
    assert gapped["data_quality"]["missing_baseline_hours"] == 3 and gapped["data_sufficient"]
    assert gapped["significance"]["result"] == SIGNIFICANT == full["significance"]["result"]
    assert "18 baseline hours" in gapped["significance"]["detail"]


def test_revenue_and_clicks_alerts_say_they_were_not_tested():
    rows = epc_history(current=[(400, 4, BASE_PAYOUT)] * 3)
    kpi = snapshot(rows, metric="revenue")
    assert kpi["significance"]["method"] == "threshold_only" and kpi["significance"]["result"] == "not_tested"
    assert kpi["statistically_significant"] is None and kpi["significant"]
    assert kpi["signal_strength"] == "moderate"  # never "strong" without a test


def test_a_significant_but_small_drop_is_not_an_incident():
    # 30 000 clicks a window: a 15% conversion-rate dip is statistically real but below the business threshold.
    kpi = snapshot(epc_history(current=[(10000, 340, BASE_PAYOUT)] * 3, clicks=10000, conversions=400))
    assert kpi["significance"]["result"] == SIGNIFICANT and kpi["metric_change_pct"] == -15.0
    assert not kpi["significant"]
