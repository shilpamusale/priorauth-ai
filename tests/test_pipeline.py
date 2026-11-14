"""A case end to end with a scripted model: decision, missing documentation, rationale and audit record."""

import datetime
import hashlib

import pytest

from priorauth.decision import decide
from priorauth.llm.base import ModelResponse, StopReason, ToolCall, Usage
from priorauth.llm.scripted import ScriptedClient
from priorauth.matcher import PROMPT_VERSION, MatchRun, StopCause, load_prompt
from priorauth.pipeline import AuditRecord, run_case
from priorauth.policies import fingerprint
from priorauth.schemas import (
    ClinicalNote,
    CriterionResult,
    CriterionStatus,
    Decision,
    Evidence,
    PARequest,
    Policy,
    PolicyCriterion,
)

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
DURATION_MET = {
    "criterion_id": "demo.duration",
    "status": "met",
    "evidence": [{"note_id": "n1", "quote": "Low back pain since March 2026"}],
    "reasoning": "Pain since March.",
}
SCALE_MET = {
    "criterion_id": "demo.scale",
    "status": "met",
    "evidence": [{"note_id": "n2", "quote": "ODI 44% at discharge"}],
    "reasoning": "ODI recorded.",
}


def submits(*findings, stop=StopReason.TOOL_USE):
    calls = [ToolCall(id=f"toolu_{i}", name="submit_finding", input=f) for i, f in enumerate(findings)]
    return ModelResponse(tool_calls=calls, stop_reason=stop, usage=Usage(input_tokens=500, output_tokens=80))


def run(*replies, max_steps=10):
    return run_case(ScriptedClient(list(replies)), POLICY, REQUEST, max_steps=max_steps)


def test_ready_case_and_its_audit_record():
    record = run(submits(DURATION_MET, SCALE_MET))
    result = record.determination
    assert result.decision is Decision.READY_TO_SUBMIT
    assert (result.case_id, result.policy_id) == ("dev_900", "demo")
    assert [r.criterion_id for r in result.criteria] == ["demo.duration", "demo.scale"]
    assert result.missing_documentation == []
    assert result.rationale.startswith("Ready to submit")
    assert record.policy_fingerprint == fingerprint(POLICY)
    assert record.prompt_version == PROMPT_VERSION
    assert record.prompt_fingerprint == hashlib.sha256(load_prompt().encode()).hexdigest()[:16]
    assert record.model == "claude-sonnet-5-5"
    assert (record.stop, record.steps, record.usage) == (
        StopCause.COMPLETE,
        1,
        Usage(input_tokens=500, output_tokens=80),
    )
    assert record.skipped == [] and record.rejected_quotes == []
    assert record.turns[1].tool_calls[0].name == "submit_finding"


def test_documentation_gap_names_what_to_add():
    gap = {"criterion_id": "demo.scale", "status": "insufficient_evidence", "evidence": [], "reasoning": ""}
    gap_with_reason = {**gap, "reasoning": "No disability scale such as ODI is recorded."}
    for finding, expected in [
        (gap_with_reason, "demo.scale: No disability scale such as ODI is recorded."),
        (gap, "demo.scale: A disability scale is recorded."),  # no reasoning: the criterion itself
    ]:
        result = run(submits(DURATION_MET, finding)).determination
        assert result.decision is Decision.NEEDS_DOCUMENTATION
        assert result.missing_documentation == [expected]
        assert result.rationale == "Needs documentation before submission. Documentation missing: demo.scale."


def test_not_met_escalates_and_says_why():
    not_met = {**DURATION_MET, "status": "not_met", "reasoning": "Pain began two weeks ago."}
    result = run(submits(not_met, SCALE_MET)).determination
    assert result.decision is Decision.ESCALATE
    assert result.rationale == "Escalated to a human reviewer. Not met: demo.duration."


def test_skipped_criteria_escalate_and_are_recorded():
    search = ModelResponse(
        tool_calls=[ToolCall(id="toolu_9", name="search_chart", input={"query": "ODI"})],
        stop_reason=StopReason.TOOL_USE,
    )
    record = run(submits(DURATION_MET), search, search, max_steps=3)
    assert record.stop is StopCause.STEP_LIMIT and record.skipped == ["demo.scale"]
    assert record.determination.decision is Decision.ESCALATE
    assert record.determination.rationale == "Escalated to a human reviewer. No finding: demo.scale."


def test_a_quote_that_got_past_the_tools_is_still_caught(monkeypatch):
    """verify() runs on the findings whatever produced them, not only on findings made through the tools."""
    invented = CriterionResult(
        criterion_id="demo.scale",
        status=CriterionStatus.MET,
        evidence=[Evidence(note_id="n2", quote="ODI 52% at intake")],
    )
    duration = CriterionResult.model_validate(DURATION_MET)
    fake_run = MatchRun(
        model="claude-sonnet-5-5",
        prompt_version=PROMPT_VERSION,
        findings=[duration, invented],
        skipped=[],
        stop=StopCause.COMPLETE,
        steps=1,
        usage=Usage(),
        turns=[],
    )
    monkeypatch.setattr("priorauth.pipeline.match", lambda *args, **kwargs: fake_run)
    record = run_case(ScriptedClient([]), POLICY, REQUEST)
    assert decide(POLICY, fake_run.findings) is Decision.READY_TO_SUBMIT  # what would happen without verify()
    assert record.determination.decision is Decision.ESCALATE
    assert record.determination.criteria == [duration]
    assert record.rejected_quotes == ["demo.scale: quote 'ODI 52% at intake' is not in n2"]
    assert record.determination.rationale == "Escalated to a human reviewer. Evidence failed verification: demo.scale."


@pytest.mark.parametrize(
    "replies",
    [
        [submits(DURATION_MET, SCALE_MET)],
        [submits({**DURATION_MET, "status": "insufficient_evidence", "evidence": []}, SCALE_MET)],
        [submits(DURATION_MET), ModelResponse(stop_reason=StopReason.OTHER)],
    ],
)
def test_decision_always_follows_the_recorded_criteria(replies):
    """The same check as the phase 4 hard gate: the recorded criteria imply the recorded decision."""
    result = run(*replies).determination
    assert decide(POLICY, result.criteria) is result.decision


def test_audit_record_survives_a_json_round_trip():
    record = run(submits(DURATION_MET, SCALE_MET))
    assert AuditRecord.model_validate_json(record.model_dump_json()) == record
    assert record.recorded_at.tzinfo is not None