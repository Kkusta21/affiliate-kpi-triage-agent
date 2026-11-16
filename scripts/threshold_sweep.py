"""EXPLORATORY threshold sensitivity on the synthetic evaluation set. Not calibration.

    python scripts/threshold_sweep.py

Moves one threshold at a time, keeps everything else at its default, and reports how the
evaluation metrics react. The defaults are NOT changed from this output, on purpose:

- The dataset is 42 synthetic cases written by the same person who wrote the rules. Choosing
  thresholds that maximise a score on it would fit the author's intuition, not real traffic.
- Many thresholds have only one or two cases near them, so a step in the table is often a
  single case flipping.

What it is good for: seeing which thresholds the current cases are sensitive to, and where a
real labelled set would need more examples. Real calibration needs real incidents (see README).
"""

from __future__ import annotations

from dataclasses import replace

from triage_agent.config import SETTINGS
from triage_agent.evaluation import evaluate

SWEEPS = {
    "drop_threshold_pct": [-20.0, -30.0, -40.0, -50.0],
    "significance_alpha": [0.05, 0.01, 0.001],
    "min_clicks_for_signal": [50, 100, 200],
    "bot_score_threshold": [0.6, 0.7, 0.8],
    "disallowed_geo_share_threshold": [0.2, 0.3, 0.4],
    "postback_ratio_threshold": [0.3, 0.5, 0.7],
    "page_latency_limit_ms": [8000, 10000, 15000],
}
COLUMNS = [
    "macro_f1",
    "primary_accuracy",
    "false_alarm_rate",
    "missed_incident_rate",
    "correct_abstention_rate",
    "unsupported_findings",
    "missed_findings",
]


def main() -> None:
    print("EXPLORATORY: synthetic cases, one threshold at a time. Defaults are marked with *. Not calibration.\n")
    print(
        f"{'threshold':<32}{'value':>8}  "
        + "  ".join(
            f"{c:>12}" for c in ["macroF1", "primary", "false_alarm", "missed_inc", "abstain_ok", "unsupported", "missed"]
        )
    )
    for name, values in SWEEPS.items():
        default = getattr(SETTINGS, name)
        for value in values:
            metrics = evaluate(settings=replace(SETTINGS, **{name: value}))["metrics"]
            cells = "  ".join(f"{'n/a' if metrics[c] is None else metrics[c]:>12}" for c in COLUMNS)
            mark = "*" if value == default else " "
            print(f"{name:<32}{value:>7}{mark}  {cells}")
        print()


if __name__ == "__main__":
    main()
