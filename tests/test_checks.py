"""Scoring an audit record against its gold case, including the hard gates."""

import datetime

import pytest

from evals.checks import run_checks
from evals.gold import GoldCase, Label
from priorauth.llm.base import ModelResponse, StopReason, ToolCall, Usage
from priorauth.llm.scripted import ScriptedClient
from priorauth.matcher import StopCause
from priorauth.pipeline import AuditRecord, run_case
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

MET, NOT_MET, GAP = CriterionStatus.MET, CriterionStatus.NOT_MET, CriterionStatus.INSUFFICIENT_EVIDENCE
POLICY = Policy(
    policy_id="demo",
    title="Demo policy",
    source="illustrative",
    criteria=[
        PolicyCriterion(criterion_id="demo.duration", text="Pain has lasted at least 3 months."),
        PolicyCriterion(criterion_id="demo.scale", text="A disability scale is recorded."),
    ],
)
REQUEST = PARequest(
    case_id="dev_900",
    payer="Medicare",
    request_date=datetime.date(2026, 10, 1),
    service_code="64493",
    icd10_codes=["M54.50"],
    notes=[
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
    ],
)
CASE = GoldCase(
    policy_id="demo",
    request=REQUEST,
    labels={
        "demo.duration": Label(status=MET, evidence_notes=["n1"]),
        "demo.scale": Label(status=MET, evidence_notes=["n2"]),
    },
    expected_decision=Decision.READY_TO_SUBMIT,
)
DURATION = CriterionResult(
    criterion_id="demo.duration", status=MET, evidence=[Evidence(note_id="n1", quote="Low back pain since March 2026")]
)
SCALE = CriterionResult(
    criterion_id="demo.scale", status=MET, evidence=[Evidence(note_id="n2", quote="ODI 44% at discharge")]
)


def record(decision, *criteria):
    """An audit record with exactly these criteria and this decision, consistent or not."""
    determination = Determination(
        case_id="dev_900",
        policy_id="demo",
        decision=decision,
        criteria=list(criteria),
        missing_documentation=["gap"] if decision is Decision.NEEDS_DOCUMENTATION else [],
        rationale="test",
    )
    return AuditRecord(
        determination=determination,
        policy_fingerprint="0" * 16,
        model="claude-sonnet-5-5",
        prompt_version="matcher_v1",
        prompt_fingerprint="0" * 16,
        stop=StopCause.COMPLETE,
        steps=1,
        usage=Usage(),
        skipped=[],
        rejected_quotes=[],
        turns=[],
    )


def check(rec, name, case=CASE, policy=POLICY):
    return {c.name: c for c in run_checks(case, policy, rec)}[name]


def failure(rec, name):
    """The detail of a check that must fail."""
    result = check(rec, name)
    assert not result.passed, name
    return result.detail


def test_a_correct_run_passes_every_check():
    results = run_checks(CASE, POLICY, record(Decision.READY_TO_SUBMIT, DURATION, SCALE))
    assert [c.name for c in results] == [
        "decision_correct",
        "criteria_correct",
        "evidence_notes_match",
        "evidence_grounded",
        "ready_requires_all_met",
        "decision_follows_criteria",
    ]
    assert all(c.passed and c.detail == "" for c in results)
    assert {c.name for c in results if c.hard_gate} == {
        "evidence_grounded",
        "ready_requires_all_met",
        "decision_follows_criteria",
    }


def test_a_real_pipeline_run_passes_every_check():
    submits = [
        ToolCall(id=f"toolu_{i}", name="submit_finding", input=r.model_dump(mode="json"))
        for i, r in enumerate([DURATION, SCALE])
    ]
    client = ScriptedClient([ModelResponse(tool_calls=submits, stop_reason=StopReason.TOOL_USE)])
    assert all(c.passed for c in run_checks(CASE, POLICY, run_case(client, POLICY, REQUEST)))


def test_wrong_decision_and_wrong_criteria_are_reported():
    gap = CriterionResult(criterion_id="demo.scale", status=GAP)
    rec = record(Decision.NEEDS_DOCUMENTATION, DURATION, gap)
    assert failure(rec, "decision_correct") == "got needs_documentation, expected ready_to_submit"
    assert failure(rec, "criteria_correct") == "demo.scale: got insufficient_evidence, expected met"


def test_a_criterion_with_no_finding_counts_as_wrong():
    result = check(record(Decision.ESCALATE, DURATION), "criteria_correct")
    assert not result.passed and result.detail == "demo.scale: got no finding, expected met"


def test_met_must_cite_a_note_the_label_names():
    elsewhere = SCALE.model_copy(update={"evidence": [Evidence(note_id="n1", quote="Started physical therapy")]})
    result = check(record(Decision.READY_TO_SUBMIT, DURATION, elsewhere), "evidence_notes_match")
    assert not result.passed and result.detail == "demo.scale: cites ['n1'], label names ['n2']"
    also_right_note = SCALE.model_copy(update={"evidence": [*elsewhere.evidence, *SCALE.evidence]})
    assert check(record(Decision.READY_TO_SUBMIT, DURATION, also_right_note), "evidence_notes_match").passed


def test_evidence_notes_are_only_compared_where_both_say_met():
    not_met = CriterionResult(
        criterion_id="demo.scale", status=NOT_MET, evidence=[Evidence(note_id="n1", quote="Started physical therapy")]
    )
    assert check(record(Decision.ESCALATE, DURATION, not_met), "evidence_notes_match").passed


def test_hard_gate_catches_a_quote_not_in_the_chart():
    invented = SCALE.model_copy(update={"evidence": [Evidence(note_id="n2", quote="ODI 52% at intake")]})
    result = check(record(Decision.READY_TO_SUBMIT, DURATION, invented), "evidence_grounded")
    assert not result.passed and result.hard_gate
    assert result.detail == "demo.scale: quote 'ODI 52% at intake' is not in n2"


def test_hard_gate_catches_ready_with_a_required_criterion_unmet():
    gap = CriterionResult(criterion_id="demo.scale", status=GAP)
    rec = record(Decision.READY_TO_SUBMIT, DURATION, gap)
    assert failure(rec, "ready_requires_all_met") == "ready_to_submit with unmet: demo.scale"
    assert failure(rec, "decision_follows_criteria") == "recorded ready_to_submit, criteria imply needs_documentation"


def test_ready_gate_ignores_optional_criteria():
    optional = PolicyCriterion(criterion_id="demo.imaging", text="Imaging is on file.", required=False)
    policy = POLICY.model_copy(update={"criteria": [*POLICY.criteria, optional]})
    labels = {**CASE.labels, "demo.imaging": Label(status=NOT_MET)}
    case = CASE.model_copy(update={"labels": labels})
    rec = record(
        Decision.READY_TO_SUBMIT, DURATION, SCALE, CriterionResult(criterion_id="demo.imaging", status=NOT_MET)
    )
    assert check(rec, "ready_requires_all_met", case, policy).passed


def test_decision_gate_catches_an_overcautious_decision_too():
    """Escalating a case its own criteria call ready is not unsafe, but the output no longer explains itself."""
    rec = record(Decision.ESCALATE, DURATION, SCALE)
    assert check(rec, "ready_requires_all_met").passed
    assert failure(rec, "decision_follows_criteria") == "recorded escalate, criteria imply ready_to_submit"


def test_a_record_from_another_case_is_refused():
    other = record(Decision.READY_TO_SUBMIT, DURATION, SCALE)
    other = other.model_copy(update={"determination": other.determination.model_copy(update={"case_id": "dev_901"})})
    with pytest.raises(ValueError, match="do not belong to case 'dev_900'"):
        run_checks(CASE, POLICY, other)