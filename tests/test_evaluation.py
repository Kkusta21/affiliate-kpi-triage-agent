"""The evaluation framework: metric definitions, undefined values, dataset hygiene, reproducibility."""

from __future__ import annotations

import json

import pytest

from triage_agent.cli import main
from triage_agent.config import REPO_ROOT
from triage_agent.evaluation import (
    NAMED_CAUSES,
    PRIMARY_LABELS,
    CaseOutcome,
    build_world,
    evaluate,
    format_report,
    load_dataset,
    score,
)

PINNED_REPORT = REPO_ROOT / "eval" / "report_offline.json"


def outcome(
    case_id, expected_primary, predicted_primary, expected=(), predicted=(), flagged=(), exp_sev=None, pred_sev=None
):
    expected, predicted = set(expected), set(predicted)
    return CaseOutcome(
        case_id,
        "test",
        expected_primary,
        predicted_primary,
        sorted(expected),
        sorted(predicted),
        sorted(flagged),
        exp_sev,
        pred_sev,
        3,
        "r",
        correct_secondary=sorted((expected - {expected_primary}) & predicted),
        unsupported=sorted(predicted - expected),
        missed=sorted(expected - predicted),
    )


# ---------------------------------------------------------------- metric definitions


def test_multi_label_counts_precision_recall_and_f1():
    m = score(
        [
            outcome(
                "a",
                "landing_page_down",
                "landing_page_down",
                ["landing_page_down", "offer_cap_reached"],
                ["landing_page_down", "offer_cap_reached"],
            ),  # both right
            outcome(
                "b",
                "offer_cap_reached",
                "offer_cap_reached",
                ["offer_cap_reached", "traffic_quality"],
                ["offer_cap_reached"],
                flagged=["traffic_quality"],
            ),  # missed a secondary
            outcome(
                "c", "tracking_broken", "tracking_broken", ["tracking_broken"], ["tracking_broken", "traffic_quality"]
            ),  # one unsupported
        ]
    )
    traffic = m["per_cause"]["traffic_quality"]
    assert (traffic["tp"], traffic["fp"], traffic["fn"]) == (0, 1, 1)
    assert traffic["precision"] == 0.0 and traffic["recall"] == 0.0 and traffic["f1"] == 0.0
    cap = m["per_cause"]["offer_cap_reached"]
    assert (cap["tp"], cap["fp"], cap["fn"], cap["f1"]) == (2, 0, 0, 1.0)
    assert m["micro_precision"] == round(4 / 5, 4) and m["micro_recall"] == round(4 / 5, 4)
    assert m["correct_secondary_findings"] == 1 and m["expected_secondary_findings"] == 2
    assert (m["unsupported_findings"], m["missed_findings"], m["missed_but_flagged"]) == (1, 1, 1)
    assert m["exact_cause_set_match"] == round(1 / 3, 4)


def test_undefined_values_are_none_not_zero_or_one():
    m = score([outcome("a", "landing_page_down", "landing_page_down", ["landing_page_down"], ["landing_page_down"])])
    paused = m["per_cause"]["offer_paused"]
    assert paused == {"support": 0, "tp": 0, "fp": 0, "fn": 0, "precision": None, "recall": None, "f1": None}
    assert "offer_paused" in m["causes_without_support"]
    assert m["macro_f1"] == 1.0  # only causes with support are averaged
    assert m["false_alarm_rate"] is None and m["correct_abstention_rate"] is None
    assert m["severity_accuracy"] is None


def test_precision_undefined_when_a_cause_is_never_predicted():
    m = score([outcome("a", "geo_mismatch", "unknown", ["geo_mismatch"], [])])
    geo = m["per_cause"]["geo_mismatch"]
    assert geo["precision"] is None and geo["recall"] == 0.0 and geo["f1"] == 0.0


def test_abstention_false_alarm_and_missed_incident_rates():
    m = score(
        [
            outcome("n1", "no_anomaly", "no_anomaly"),
            outcome("n2", "no_anomaly", "unknown"),  # false alarm (an incident is opened)
            outcome("u1", "unknown", "unknown"),
            outcome("u2", "unknown", "no_anomaly"),  # failed to abstain, and a missed incident
            outcome("i1", "landing_page_down", "unknown", ["landing_page_down"]),  # abstained wrongly
        ]
    )
    assert m["false_alarm_rate"] == 0.5
    assert m["correct_abstention_rate"] == 0.5 and m["abstention_precision"] == round(1 / 3, 4)
    assert m["missed_incident_rate"] == round(1 / 3, 4)
    assert m["confusion_matrix"]["no_anomaly"] == {"no_anomaly": 1, "unknown": 1}
    assert m["primary_accuracy"] == 0.4


def test_severity_is_only_scored_where_labelled():
    m = score(
        [
            outcome("a", "landing_page_down", "landing_page_down", exp_sev="critical", pred_sev="critical"),
            outcome("b", "offer_cap_reached", "offer_cap_reached", exp_sev="medium", pred_sev="low"),
            outcome("c", "unknown", "unknown", pred_sev="high"),
        ]
    )
    assert m["severity_accuracy"] == 0.5 and m["severity_labelled_cases"] == 2


# ---------------------------------------------------------------- dataset hygiene


def test_dataset_is_marked_synthetic_and_every_case_is_labelled_and_explained():
    dataset = load_dataset()
    assert dataset["synthetic"] is True and "Not production data" in dataset["description"]
    ids = [c["id"] for c in dataset["cases"]]
    assert len(ids) == len(set(ids)) >= 40
    for case in dataset["cases"]:
        expected = case["expected"]
        assert expected["primary"] in PRIMARY_LABELS and len(case["rationale"]) > 40, case["id"]
        assert set(expected["causes"]) <= set(NAMED_CAUSES), case["id"]
        if expected["primary"] in NAMED_CAUSES:
            assert expected["primary"] in expected["causes"], case["id"]
        else:
            assert expected["causes"] == [], case["id"]
        world = build_world(case)
        assert case["alert"]["campaign_id"] in world["campaigns"]


def test_dataset_covers_the_required_scenarios():
    categories = {c["category"] for c in load_dataset()["cases"]}
    assert {
        "normal_variation",
        "landing_page_unavailable",
        "landing_page_slow",
        "tracking",
        "offer",
        "traffic",
        "epc",
        "seasonal",
        "insufficient_evidence",
        "multiple_causes",
        "conflicting_evidence",
    } <= categories


def test_eval_campaigns_are_separate_from_the_demo_samples():
    demo = json.loads((REPO_ROOT / "data" / "world.json").read_text())["campaigns"]
    assert not {c["campaign"]["id"] for c in load_dataset()["cases"]} & set(demo)


# ---------------------------------------------------------------- reproducibility


@pytest.fixture(scope="module")
def report():
    return evaluate()


def test_evaluation_is_deterministic(report):
    assert evaluate() == report


def test_offline_results_match_the_pinned_report(report):
    """The committed report is the reference. If a code change moves any metric or any case,
    this fails, and the report must be regenerated on purpose:
        python -m triage_agent evaluate --json eval/report_offline.json"""
    pinned = json.loads(PINNED_REPORT.read_text())
    assert report["metrics"] == pinned["metrics"]
    assert report["cases"] == pinned["cases"]


def test_report_text_states_the_data_is_synthetic(report):
    text = format_report(report)
    assert "SYNTHETIC" in text and "Not production data" in text and "macro F1" in text


def test_cli_evaluate_writes_the_json_report(tmp_path, capsys):
    out = tmp_path / "report.json"
    assert main(["evaluate", "--json", str(out)]) == 0
    assert json.loads(out.read_text())["metrics"]["cases"] >= 40
    assert "primary root cause accuracy" in capsys.readouterr().out
