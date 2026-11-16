"""Metrics across trials and the CI gate rule."""

import datetime

import pytest

from evals.checks import run_checks
from evals.gold import GoldCase, Label
from evals.runner import TrialResult
from evals.scoring import format_summary, gate, summarize
from priorauth.llm.base import Usage
from priorauth.matcher import StopCause
from priorauth.pipeline import AuditRecord
from priorauth.schemas import (
    ClinicalNote,
    CriterionResult,
    CriterionStatus,
    Decision,
    Determination,
    Evidence,
    PARequest,
    Policy,
    PolicyCriterion,
)

MET, GAP = CriterionStatus.MET, CriterionStatus.INSUFFICIENT_EVIDENCE
READY, DOCS, ESCALATE = Decision.READY_TO_SUBMIT, Decision.NEEDS_DOCUMENTATION, Decision.ESCALATE
POLICY = Policy(
    policy_id="demo",
    title="Demo policy",
    source="illustrative",
    service_codes=["64493"],
    criteria=[
        PolicyCriterion(criterion_id="demo.duration", text="Pain has lasted at least 3 months."),
        PolicyCriterion(criterion_id="demo.scale", text="A disability scale is recorded."),
    ],
)
NOTES = [
    ClinicalNote(
        note_id="n1",
        date=datetime.date(2026, 6, 1),
        note_type="primary care visit",
        text="Low back pain since March 2026. Started physical therapy.",
    ),
    ClinicalNote(
        note_id="n2",
        date=datetime.date(2026, 9, 1),
        note_type="physical therapy discharge summary",
        text="ODI 44% at discharge. No change with physical therapy.",
    ),
]
DURATION = CriterionResult(
    criterion_id="demo.duration", status=MET, evidence=[Evidence(note_id="n1", quote="Low back pain since March 2026")]
)
SCALE = CriterionResult(
    criterion_id="demo.scale", status=MET, evidence=[Evidence(note_id="n2", quote="ODI 44% at discharge")]
)
SCALE_GAP = CriterionResult(criterion_id="demo.scale", status=GAP, reasoning="No disability scale.")


def gold(case_id, scale, decision):
    request = PARequest(
        case_id=case_id,
        payer="Medicare",
        request_date=datetime.date(2026, 10, 1),
        service_code="64493",
        icd10_codes=["M54.50"],
        notes=NOTES,
    )
    labels = {"demo.duration": Label(status=MET, evidence_notes=["n1"]), "demo.scale": scale}
    return GoldCase(policy_id="demo", request=request, labels=labels, expected_decision=decision)


CASES = {
    "dev_900": gold("dev_900", Label(status=MET, evidence_notes=["n2"]), READY),
    "dev_901": gold("dev_901", Label(status=GAP), DOCS),
}


def trial(case_id, number, decision, *criteria, model="claude-sonnet-5-5"):
    determination = Determination(
        case_id=case_id,
        policy_id="demo",
        decision=decision,
        criteria=list(criteria),
        missing_documentation=["gap"] if decision is DOCS else [],
        rationale="test",
    )
    record = AuditRecord(
        determination=determination,
        policy_fingerprint="0" * 16,
        model=model,
        prompt_version="matcher_v1",
        prompt_fingerprint="0" * 16,
        stop=StopCause.COMPLETE,
        steps=2,
        usage=Usage(input_tokens=10_000, output_tokens=1_000),
        skipped=[],
        rejected_quotes=[],
        turns=[],
    )
    checks = run_checks(CASES[case_id], POLICY, record)
    return TrialResult(case_id=case_id, trial=number, checks=checks, record=record)


def correct_trials():
    return [
        trial("dev_900", 1, READY, DURATION, SCALE),
        trial("dev_900", 2, READY, DURATION, SCALE),
        trial("dev_901", 1, DOCS, DURATION, SCALE_GAP),
        trial("dev_901", 2, DOCS, DURATION, SCALE_GAP),
    ]


def test_correct_trials_score_perfectly_and_pass_the_gate():
    summary = summarize(correct_trials(), CASES)
    assert (summary.cases, summary.trials, summary.errors) == (2, 4, 0)
    assert summary.decision_accuracy == 1.0 and summary.criterion_accuracy == 1.0
    assert set(summary.check_pass_rates.values()) == {1.0} and len(summary.check_pass_rates) == 6
    assert summary.hard_gate_failures == [] and summary.unstable_cases == []
    assert (summary.input_tokens, summary.output_tokens) == (40_000, 4_000)
    assert summary.cost_usd == pytest.approx(0.12)  # 40k x $2/M + 4k x $10/M
    assert gate(summary, min_decision_accuracy=0.9) == []


def test_a_wrong_trial_lowers_accuracy_and_marks_the_case_unstable():
    trials = correct_trials()
    trials[3] = trial("dev_901", 2, ESCALATE, DURATION)  # scale skipped
    summary = summarize(trials, CASES)
    assert summary.decision_accuracy == 0.75
    assert summary.criterion_accuracy == 7 / 8  # the skipped criterion counts as wrong
    assert summary.check_pass_rates["decision_correct"] == 0.75
    assert summary.unstable_cases == ["dev_901"]


def test_a_hard_gate_failure_fails_the_gate_whatever_the_accuracy():
    trials = correct_trials()
    trials[2] = trial("dev_901", 1, READY, DURATION, SCALE_GAP)  # ready with a gap
    summary = summarize(trials, CASES)
    assert summary.hard_gate_failures == [
        "dev_901 trial 1: ready_requires_all_met: ready_to_submit with unmet: demo.scale",
        "dev_901 trial 1: decision_follows_criteria: recorded ready_to_submit, criteria imply needs_documentation",
    ]
    assert gate(summary, min_decision_accuracy=0.0)[0].startswith("hard gate: dev_901 trial 1")


def test_trials_without_a_record_are_counted_apart_and_fail_the_gate():
    trials = [*correct_trials()[:2], TrialResult(case_id="dev_901", trial=1, error="no recording")]
    summary = summarize(trials, CASES)
    assert (summary.cases, summary.trials, summary.errors) == (2, 3, 1)
    assert summary.decision_accuracy == 1.0  # over the two trials that ran
    assert summary.check_pass_rates["decision_correct"] == 1.0
    assert gate(summary, min_decision_accuracy=0.0) == ["1 trials have no record"]


def test_accuracy_below_the_threshold_fails_the_gate():
    trials = correct_trials()
    trials[0] = trial("dev_900", 1, ESCALATE, DURATION)
    assert gate(summarize(trials, CASES), min_decision_accuracy=0.8) == ["decision accuracy 0.75 is below 0.80"]


def test_no_records_at_all():
    summary = summarize([TrialResult(case_id="dev_900", trial=1, error="no recording")], CASES)
    assert summary.decision_accuracy == 0.0 and summary.check_pass_rates == {} and summary.cost_usd == 0.0


def test_cost_is_unknown_for_a_model_without_a_price():
    trials = [*correct_trials()[:1], trial("dev_901", 1, DOCS, DURATION, SCALE_GAP, model="some-other-model")]
    summary = summarize(trials, CASES)
    assert summary.cost_usd is None
    assert "cost unknown" in format_summary(summary)


def test_summary_text():
    trials = correct_trials()
    trials[3] = trial("dev_901", 2, ESCALATE, DURATION)
    assert format_summary(summarize(trials, CASES)).splitlines() == [
        "2 cases, 4 trials, 0 errors",
        "decision accuracy 0.75, criterion accuracy 0.88",
        "checks: decision_correct 0.75, criteria_correct 0.75, evidence_notes_match 1.00, evidence_grounded 1.00, "
        "ready_requires_all_met 1.00, decision_follows_criteria 1.00",
        "unstable cases: dev_901",
        "hard-gate failures: 0",
        "tokens: 40,000 in, 4,000 out, cost $0.12",
    ]