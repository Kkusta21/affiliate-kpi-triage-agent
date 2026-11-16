"""Offline evaluation against the labelled SYNTHETIC dataset in eval/dataset.json.

    python -m triage_agent evaluate                 # offline policy, deterministic
    python -m triage_agent evaluate --json out.json # also write the full report
    python -m triage_agent evaluate --llm anthropic:claude-haiku-5-5   # same cases, live planner

Each case gets its own small world (campaign, offer, KPI history, page, tracking, traffic),
a fresh in-memory incident store and no sleeps, so a run is fully reproducible.

What is scored
--------------
Multi-label: an incident can have several causes. The PREDICTED causes of a case are its
CONFIRMED findings. Suspected and inconclusive findings are not predictions: they are tracked
separately as "flagged but unconfirmed", because the agent explicitly did not commit to them.

- Per named cause: precision, recall and F1 from TP/FP/FN over cases. A value whose
  denominator is zero is undefined and reported as None, never as 0 or 1.
  F1 = 2TP / (2TP + FP + FN), defined whenever the cause was expected or predicted at all.
- Macro F1 over causes with at least one expected case. Micro precision, recall and F1.
- Primary accuracy: the predicted root cause (including no_anomaly and unknown) equals the label.
- Abstention: among cases labelled "unknown", the share where the agent also said unknown,
  and among cases where the agent said unknown, the share that were labelled unknown.
- False alarm rate: among cases labelled no_anomaly, the share where an incident was opened.
- Missed incident rate: among cases labelled with a real incident, the share called no_anomaly.
- Severity accuracy on the cases where a severity is labelled.
- A confusion matrix of expected vs predicted primary.
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from .agent import run_triage
from .config import REPO_ROOT, SETTINGS, Settings
from .datasource import DataSource
from .llm import LLM, OfflineLLM
from .models import FindingStatus, RootCause
from .synthetic import DEFAULT_END, hourly_history
from .tools.actions import IncidentStore
from .tracing import Tracer

DEFAULT_DATASET = REPO_ROOT / "eval" / "dataset.json"
NAMED_CAUSES = [c.value for c in RootCause if c not in (RootCause.no_anomaly, RootCause.unknown)]
PRIMARY_LABELS = NAMED_CAUSES + [RootCause.no_anomaly.value, RootCause.unknown.value]


# ---------------------------------------------------------------- running the cases


def build_world(case: dict[str, Any]) -> dict[str, Any]:
    """Turn one dataset case into the dict a DataSource serves."""
    campaign, offer = case["campaign"], case["offer"]
    spec = case["kpi"]["generator"]
    rows: list[dict[str, Any]] = []
    if spec is not None:
        end = datetime.fromisoformat(spec["end"].replace("Z", "+00:00")) if spec.get("end") else DEFAULT_END
        rows = hourly_history(
            spec["base_clicks"],
            spec["base_cr"],
            spec["payout"],
            spec.get("current"),
            spec.get("days", 8),
            spec.get("weekend_factor", 1.0),
            spec.get("drop", ()),
            end,
            spec.get("payout_pattern", False),
        )
    cid = campaign["id"]
    return {
        "campaigns": {cid: campaign},
        "offers": {offer["id"]: offer},
        "kpi": {cid: {"hourly": rows}},
        "landing_pages": {campaign["landing_page_url"]: {"responses": case["page_responses"]}},
        "tracking": {cid: case["tracking"]},
        "traffic": {cid: case["traffic"]},
    }


@dataclass
class CaseOutcome:
    id: str
    category: str
    expected_primary: str
    predicted_primary: str
    expected_causes: list[str]
    predicted_causes: list[str]  # confirmed findings
    flagged_unconfirmed: list[str]  # suspected or inconclusive findings
    expected_severity: str | None
    predicted_severity: str | None
    steps: int
    rationale: str
    correct_secondary: list[str] = field(default_factory=list)
    unsupported: list[str] = field(default_factory=list)
    missed: list[str] = field(default_factory=list)

    @property
    def primary_correct(self) -> bool:
        return self.expected_primary == self.predicted_primary


def run_case(case: dict[str, Any], llm: LLM | None = None, settings: Settings = SETTINGS) -> CaseOutcome:
    result = run_triage(
        case["alert"],
        llm=llm or OfflineLLM(settings),
        ds=DataSource(world=build_world(case)),
        incidents=IncidentStore(":memory:"),
        tracer=Tracer(None),
        settings=settings,
        sleep=lambda _: None,
    )
    if result.status != "completed":
        predicted_primary, confirmed, flagged, severity = f"<{result.status}>", set(), set(), None
    else:
        assert result.root_cause is not None
        predicted_primary = result.root_cause.value
        confirmed = {f.cause.value for f in result.findings if f.status is FindingStatus.confirmed}
        flagged = {f.cause.value for f in result.findings if f.status is not FindingStatus.confirmed}
        severity = result.severity.value if result.severity else None
    expected = set(case["expected"]["causes"])
    primary = case["expected"]["primary"]
    return CaseOutcome(
        id=case["id"],
        category=case["category"],
        expected_primary=primary,
        predicted_primary=predicted_primary,
        expected_causes=sorted(expected),
        predicted_causes=sorted(confirmed),
        flagged_unconfirmed=sorted(flagged),
        expected_severity=case["expected"].get("severity"),
        predicted_severity=severity,
        steps=len(result.steps),
        rationale=case["rationale"],
        correct_secondary=sorted((expected - {primary}) & confirmed),
        unsupported=sorted(confirmed - expected),
        missed=sorted(expected - confirmed),
    )


# ---------------------------------------------------------------- scoring


def _ratio(numerator: int, denominator: int) -> float | None:
    return round(numerator / denominator, 4) if denominator else None


def score(outcomes: list[CaseOutcome]) -> dict[str, Any]:
    per_cause: dict[str, dict[str, Any]] = {}
    totals = Counter[str]()
    for cause in NAMED_CAUSES:
        tp = sum(cause in o.expected_causes and cause in o.predicted_causes for o in outcomes)
        fp = sum(cause not in o.expected_causes and cause in o.predicted_causes for o in outcomes)
        fn = sum(cause in o.expected_causes and cause not in o.predicted_causes for o in outcomes)
        totals.update(tp=tp, fp=fp, fn=fn)
        per_cause[cause] = {
            "support": tp + fn,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "precision": _ratio(tp, tp + fp),
            "recall": _ratio(tp, tp + fn),
            "f1": _ratio(2 * tp, 2 * tp + fp + fn),
        }
    with_support = [c for c, m in per_cause.items() if m["support"] > 0]
    f1s = [per_cause[c]["f1"] for c in with_support]
    tp, fp, fn = totals["tp"], totals["fp"], totals["fn"]

    expected_unknown = [o for o in outcomes if o.expected_primary == "unknown"]
    predicted_unknown = [o for o in outcomes if o.predicted_primary == "unknown"]
    normal = [o for o in outcomes if o.expected_primary == "no_anomaly"]
    incidents = [o for o in outcomes if o.expected_primary != "no_anomaly"]
    with_severity = [o for o in outcomes if o.expected_severity]

    confusion: dict[str, dict[str, int]] = {}
    for o in outcomes:
        row = confusion.setdefault(o.expected_primary, {})
        row[o.predicted_primary] = row.get(o.predicted_primary, 0) + 1

    return {
        "cases": len(outcomes),
        "per_cause": per_cause,
        "causes_without_support": [c for c in NAMED_CAUSES if c not in with_support],
        "macro_f1": round(sum(f1s) / len(f1s), 4) if f1s else None,
        "micro_precision": _ratio(tp, tp + fp),
        "micro_recall": _ratio(tp, tp + fn),
        "micro_f1": _ratio(2 * tp, 2 * tp + fp + fn),
        "primary_accuracy": _ratio(sum(o.primary_correct for o in outcomes), len(outcomes)),
        "exact_cause_set_match": _ratio(sum(o.expected_causes == o.predicted_causes for o in outcomes), len(outcomes)),
        "correct_abstention_rate": _ratio(
            sum(o.predicted_primary == "unknown" for o in expected_unknown), len(expected_unknown)
        ),
        "abstention_precision": _ratio(
            sum(o.expected_primary == "unknown" for o in predicted_unknown), len(predicted_unknown)
        ),
        "false_alarm_rate": _ratio(sum(o.predicted_primary != "no_anomaly" for o in normal), len(normal)),
        "missed_incident_rate": _ratio(sum(o.predicted_primary == "no_anomaly" for o in incidents), len(incidents)),
        "severity_accuracy": _ratio(
            sum(o.expected_severity == o.predicted_severity for o in with_severity), len(with_severity)
        ),
        "severity_labelled_cases": len(with_severity),
        "correct_secondary_findings": sum(len(o.correct_secondary) for o in outcomes),
        "expected_secondary_findings": sum(len(set(o.expected_causes) - {o.expected_primary}) for o in outcomes),
        "unsupported_findings": sum(len(o.unsupported) for o in outcomes),
        "missed_findings": sum(len(o.missed) for o in outcomes),
        "missed_but_flagged": sum(len(set(o.missed) & set(o.flagged_unconfirmed)) for o in outcomes),
        "avg_tool_calls": round(sum(o.steps for o in outcomes) / len(outcomes), 2) if outcomes else None,
        "confusion_matrix": confusion,
    }


def load_dataset(path: Path = DEFAULT_DATASET) -> dict[str, Any]:
    dataset: dict[str, Any] = json.loads(path.read_text())
    return dataset


def evaluate(path: Path = DEFAULT_DATASET, llm: LLM | None = None, settings: Settings = SETTINGS) -> dict[str, Any]:
    dataset = load_dataset(path)
    outcomes = [run_case(case, llm, settings) for case in dataset["cases"]]
    return {
        "dataset": {
            "path": str(path.relative_to(REPO_ROOT)) if path.is_relative_to(REPO_ROOT) else str(path),
            "synthetic": dataset.get("synthetic", True),
            "cases": len(outcomes),
            "categories": dict(sorted(Counter(o.category for o in outcomes).items())),
        },
        "planner": llm.mode if llm else "offline",
        "metrics": score(outcomes),
        "cases": [{**asdict(o), "primary_correct": o.primary_correct} for o in outcomes],
    }


# ---------------------------------------------------------------- report


def _fmt(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.2f}"


def format_report(report: dict[str, Any]) -> str:
    m = report["metrics"]
    d = report["dataset"]
    lines = [
        f"Evaluation on {d['cases']} SYNTHETIC labelled cases ({d['path']}), planner: {report['planner']}",
        "Not production data. Labels follow the analyst policy in scripts/make_eval_dataset.py.",
        "",
        f"{'cause':<20}{'support':>8}{'TP':>5}{'FP':>5}{'FN':>5}{'precision':>11}{'recall':>8}{'F1':>7}",
    ]
    for cause, c in m["per_cause"].items():
        lines.append(
            f"{cause:<20}{c['support']:>8}{c['tp']:>5}{c['fp']:>5}{c['fn']:>5}"
            f"{_fmt(c['precision']):>11}{_fmt(c['recall']):>8}{_fmt(c['f1']):>7}"
        )
    lines += [
        "",
        f"macro F1 (causes with support)  {_fmt(m['macro_f1'])}",
        f"micro precision / recall / F1   {_fmt(m['micro_precision'])} / {_fmt(m['micro_recall'])} / {_fmt(m['micro_f1'])}",
        f"primary root cause accuracy     {_fmt(m['primary_accuracy'])}",
        f"exact cause-set match           {_fmt(m['exact_cause_set_match'])}",
        f"correct abstention rate         {_fmt(m['correct_abstention_rate'])}   (labelled unknown -> said unknown)",
        f"abstention precision            {_fmt(m['abstention_precision'])}   (said unknown -> labelled unknown)",
        f"false alarm rate                {_fmt(m['false_alarm_rate'])}   (labelled no_anomaly -> incident opened)",
        f"missed incident rate            {_fmt(m['missed_incident_rate'])}   (real incident -> called no_anomaly)",
        f"severity accuracy               {_fmt(m['severity_accuracy'])}   ({m['severity_labelled_cases']} labelled)",
        f"secondary findings found        {m['correct_secondary_findings']} of {m['expected_secondary_findings']}",
        f"unsupported / missed findings   {m['unsupported_findings']} / {m['missed_findings']} "
        f"({m['missed_but_flagged']} of the missed were flagged as suspected or inconclusive)",
        f"average tool calls per case     {m['avg_tool_calls']}",
        "",
        "Primary confusion (rows = expected, columns = predicted, only non-zero cells):",
    ]
    for expected, row in sorted(m["confusion_matrix"].items()):
        cells = ", ".join(f"{p}: {n}" for p, n in sorted(row.items()))
        lines.append(f"  {expected:<20} {cells}")
    wrong = [c for c in report["cases"] if not c["primary_correct"] or c["missed"] or c["unsupported"]]
    if wrong:
        lines += ["", "Cases with an error:"]
        for c in wrong:
            parts = []
            if not c["primary_correct"]:
                parts.append(f"primary {c['predicted_primary']} (expected {c['expected_primary']})")
            if c["missed"]:
                flagged = set(c["missed"]) & set(c["flagged_unconfirmed"])
                parts.append(
                    "missed "
                    + ", ".join(c["missed"])
                    + (f" [flagged as unconfirmed: {', '.join(sorted(flagged))}]" if flagged else "")
                )
            if c["unsupported"]:
                parts.append("unsupported " + ", ".join(c["unsupported"]))
            lines.append(f"  {c['id']:<32} " + "; ".join(parts))
    return "\n".join(lines)
