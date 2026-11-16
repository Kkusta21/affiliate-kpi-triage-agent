"""Statistical tests for the KPI snapshot. Dependency free, deterministic.

Every test asks one question: is the CURRENT window worse than the BASELINE window by more
than chance? All tests are one-sided (a drop), at the level `alpha` from config.

Conversion rate (conversions / clicks)
    Each click converts or not, so a conversion rate is a binomial proportion.
    - Two-proportion z test when the normal approximation is reasonable: at least 5 expected
      successes and 5 expected failures in each window under the pooled rate.
    - Otherwise Fisher's exact test (hypergeometric lower tail), which needs no approximation.
    Limits: clicks are assumed independent. Real traffic is clustered (one bad source, one hour
    with a broken deploy), so the true variance is larger and these p-values are optimistic.

EPC (revenue / clicks)
    EPC is not a proportion. It is conversion rate times payout per conversion, and payouts
    are skewed (a few large orders). A binomial test would miss a payout cut completely.
    We use a parametric bootstrap that keeps both sources of variation:
      1. resample the baseline HOURS with replacement (captures hour-to-hour variation and
         the uncertainty of the baseline itself),
      2. from that resample take the conversion rate and the payout distribution,
      3. simulate the current window: conversions ~ Binomial(current clicks, rate), and each
         conversion's payout drawn from the resampled payouts,
      4. the p-value is the share of simulated current EPCs at or below the observed one.
    The middle 95% of the simulated EPCs is reported as the range expected under the baseline.
    Limits: only hourly aggregates exist, so a "payout" is an hour's average payout per
    conversion. Within-hour skew is lost, which makes the test somewhat overconfident.
    With conversion-level revenue you would resample individual payouts instead.

Revenue per hour and clicks per hour
    No test. Volume moves with time of day and with the affiliate's buying, and an hour count
    is not a proportion. These alerts are judged on the business threshold and volume only,
    and the result says so (method "threshold_only").
"""

from __future__ import annotations

import math
import random
from dataclasses import asdict, dataclass
from typing import Any, Sequence

SIGNIFICANT, NOT_SIGNIFICANT, INCONCLUSIVE, NOT_TESTED = "significant", "not_significant", "inconclusive", "not_tested"
BOOTSTRAP_SEED = 20261116  # fixed: the same data always gives the same p-value


@dataclass(frozen=True)
class TestResult:
    metric: str
    method: str  # two_proportion_z | fisher_exact | parametric_bootstrap | threshold_only
    result: str  # significant | not_significant | inconclusive | not_tested
    p_value: float | None = None
    alpha: float | None = None
    statistic: float | None = None  # the z score, for the z test
    observed: float | None = None  # current value of the metric
    expected_low: float | None = None  # bootstrap: 2.5th percentile of the current value under the baseline
    expected_high: float | None = None  # bootstrap: 97.5th percentile
    detail: str = ""

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def normal_cdf(x: float) -> float:
    return 0.5 * math.erfc(-x / math.sqrt(2))


def _log_comb(n: int, k: int) -> float:
    return math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1)


def fisher_lower_tail(cur_conv: int, cur_clicks: int, base_conv: int, base_clicks: int) -> float:
    """P(current conversions <= observed) given the margins, under "same rate in both windows"."""
    total, successes, drawn = cur_clicks + base_clicks, cur_conv + base_conv, cur_clicks
    low = max(0, drawn - (total - successes))
    denominator = _log_comb(total, drawn)
    tail = sum(
        math.exp(_log_comb(successes, x) + _log_comb(total - successes, drawn - x) - denominator)
        for x in range(low, cur_conv + 1)
    )
    return min(1.0, tail)


def _decide(p: float, alpha: float) -> str:
    return SIGNIFICANT if p < alpha else NOT_SIGNIFICANT


def conversion_rate_test(base_conv: int, base_clicks: int, cur_conv: int, cur_clicks: int, alpha: float) -> TestResult:
    """One-sided test that the current conversion rate is lower than the baseline rate."""
    observed = round(cur_conv / cur_clicks, 4) if cur_clicks else None
    if base_clicks <= 0 or cur_clicks <= 0:
        return TestResult(
            "conversion_rate",
            "none",
            INCONCLUSIVE,
            alpha=alpha,
            observed=observed,
            detail="one of the windows has no clicks",
        )
    if base_conv + cur_conv == 0:
        return TestResult(
            "conversion_rate",
            "none",
            INCONCLUSIVE,
            alpha=alpha,
            observed=observed,
            detail="no conversions in either window, so there is no rate to compare",
        )
    pooled = (base_conv + cur_conv) / (base_clicks + cur_clicks)
    smallest_expected = min(n * share for n in (base_clicks, cur_clicks) for share in (pooled, 1 - pooled))
    if smallest_expected >= 5:
        se = math.sqrt(pooled * (1 - pooled) * (1 / base_clicks + 1 / cur_clicks))
        z = (cur_conv / cur_clicks - base_conv / base_clicks) / se
        p = normal_cdf(z)
        return TestResult(
            "conversion_rate", "two_proportion_z", _decide(p, alpha), round(p, 6), alpha, round(z, 2), observed
        )
    p = fisher_lower_tail(cur_conv, cur_clicks, base_conv, base_clicks)
    return TestResult(
        "conversion_rate",
        "fisher_exact",
        _decide(p, alpha),
        round(p, 6),
        alpha,
        None,
        observed,
        detail=f"expected counts below 5 ({smallest_expected:.1f}), so the exact test is used",
    )


def _binomial(rng: random.Random, n: int, p: float) -> int:
    """Exact binomial draw by geometric jumps between successes: O(number of successes)."""
    if p <= 0.0 or n <= 0:
        return 0
    if p >= 1.0:
        return n
    log_q = math.log1p(-p)
    successes, position = 0, 0
    while True:
        position += int(math.log(1.0 - rng.random()) / log_q) + 1
        if position > n:
            return successes
        successes += 1


def epc_bootstrap_test(
    baseline_hours: Sequence[tuple[int, int, float]],
    cur_clicks: int,
    cur_revenue: float,
    alpha: float,
    min_baseline_conversions: int = 20,
    draws: int = 2000,
    seed: int = BOOTSTRAP_SEED,
) -> TestResult:
    """One-sided test that the current EPC is lower than the baseline would produce.

    `baseline_hours` holds (clicks, conversions, revenue) per baseline hour.
    """
    observed = round(cur_revenue / cur_clicks, 4) if cur_clicks else None
    hours = [h for h in baseline_hours if h[0] > 0]
    base_clicks = sum(h[0] for h in hours)
    base_conv = sum(h[1] for h in hours)
    base_revenue = sum(h[2] for h in hours)
    if cur_clicks <= 0 or base_clicks <= 0:
        return TestResult(
            "epc",
            "parametric_bootstrap",
            INCONCLUSIVE,
            alpha=alpha,
            observed=observed,
            detail="one of the windows has no clicks",
        )
    if base_conv < min_baseline_conversions:
        return TestResult(
            "epc",
            "parametric_bootstrap",
            INCONCLUSIVE,
            alpha=alpha,
            observed=observed,
            detail=f"only {base_conv} baseline conversions, fewer than the {min_baseline_conversions} "
            "needed to estimate the payout distribution",
        )
    if base_revenue <= 0:
        return TestResult(
            "epc",
            "parametric_bootstrap",
            INCONCLUSIVE,
            alpha=alpha,
            observed=observed,
            detail="the baseline has no revenue, so there is no EPC to fall from",
        )

    rng = random.Random(seed)
    simulated: list[float] = []
    for _ in range(draws):
        sample = [hours[rng.randrange(len(hours))] for _ in hours]
        clicks = sum(h[0] for h in sample)
        conversions = sum(h[1] for h in sample)
        rate = conversions / clicks if clicks else 0.0
        payouts = [h[2] / h[1] for h in sample if h[1] > 0]
        weights = [h[1] for h in sample if h[1] > 0]
        k = _binomial(rng, cur_clicks, rate)
        revenue = sum(rng.choices(payouts, weights=weights, k=k)) if k and payouts else 0.0
        simulated.append(revenue / cur_clicks)
    simulated.sort()
    current_epc = cur_revenue / cur_clicks
    at_or_below = sum(1 for value in simulated if value <= current_epc + 1e-12)
    p = (1 + at_or_below) / (draws + 1)
    return TestResult(
        "epc",
        "parametric_bootstrap",
        _decide(p, alpha),
        round(p, 6),
        alpha,
        None,
        observed,
        expected_low=round(simulated[int(0.025 * draws)], 4),
        expected_high=round(simulated[int(0.975 * draws) - 1], 4),
        detail=f"{draws} draws, resampling {len(hours)} baseline hours",
    )


def threshold_only(metric: str, observed: float | None) -> TestResult:
    return TestResult(
        metric,
        "threshold_only",
        NOT_TESTED,
        observed=observed,
        detail="no statistical test for this metric; judged on the business threshold and volume",
    )
