"""The documentation agent: one request per documentation gap, checked before it is recorded."""

import datetime
import re
from pathlib import Path

import pytest

from priorauth.documentation import PROMPT_VERSION, gaps, write_requests
from priorauth.llm.base import ModelResponse, StopReason, ToolCall, Usage
from priorauth.llm.scripted import ScriptedClient
from priorauth.matcher import StopCause, load_prompt
from priorauth.policies import load_policies
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

POLICY = Policy(
    policy_id="demo",
    title="Demo policy",
    source="illustrative",
    service_codes=["64493"],
    criteria=[
        PolicyCriterion(criterion_id="demo.duration", text="Pain has lasted at least 3 months."),
        PolicyCriterion(criterion_id="demo.scale", text="A disability scale is recorded."),
        PolicyCriterion(criterion_id="demo.therapy", text="Physical therapy was tried and failed."),
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
            text="Low back pain since March 2026. Patient says she did some therapy last year.",
        ),
    ],
)
MET = CriterionResult(
    criterion_id="demo.duration",
    status=CriterionStatus.MET,
    evidence=[Evidence(note_id="n1", quote="Low back pain since March 2026")],
)
SCALE_GAP = CriterionResult(
    criterion_id="demo.scale", status=CriterionStatus.INSUFFICIENT_EVIDENCE, reasoning="No disability scale recorded."
)
THERAPY_GAP = CriterionResult(
    criterion_id="demo.therapy", status=CriterionStatus.INSUFFICIENT_EVIDENCE, reasoning="Therapy is vague."
)


def determination(*criteria):
    has_gap = any(c.status is CriterionStatus.INSUFFICIENT_EVIDENCE for c in criteria)
    return Determination(
        case_id="dev_900",
        policy_id="demo",
        decision=Decision.NEEDS_DOCUMENTATION if has_gap else Decision.READY_TO_SUBMIT,
        criteria=list(criteria),
        missing_documentation=["gap"] if has_gap else [],
        rationale="test",
    )


SCALE_REQUEST = {
    "criterion_id": "demo.scale",
    "request": "Please record a baseline disability scale such as the Oswestry Disability Index at the next visit.",
    "note_type": "next visit note",
}
THERAPY_REQUEST = {
    "criterion_id": "demo.therapy",
    "request": "Please document the dates, length and outcome of the physical therapy the patient mentions.",
    "note_type": "therapy summary or visit note",
}


def writes(*requests, stop=StopReason.TOOL_USE):
    calls = [ToolCall(id=f"toolu_{i}", name="write_request", input=r) for i, r in enumerate(requests)]
    return ModelResponse(tool_calls=calls, stop_reason=stop, usage=Usage(input_tokens=800, output_tokens=150))


def run(*replies, criteria=(MET, SCALE_GAP, THERAPY_GAP)):
    model = ScriptedClient(list(replies))
    return write_requests(model, POLICY, REQUEST, determination(*criteria)), model


def test_gaps_are_the_insufficient_criteria_in_policy_order():
    assert gaps(determination(MET, SCALE_GAP, THERAPY_GAP)) == [SCALE_GAP, THERAPY_GAP]


def test_no_gaps_means_no_model_call():
    result, model = run(criteria=(MET,))
    assert result is None and model.requests == []


def test_one_request_per_gap_in_policy_order():
    result, model = run(writes(THERAPY_REQUEST, SCALE_REQUEST))
    assert [r.criterion_id for r in result.requests] == ["demo.scale", "demo.therapy"]
    assert result.requests[0].note_type == "next visit note"
    assert (result.stop, result.steps, result.skipped) == (StopCause.COMPLETE, 1, [])
    assert result.usage == Usage(input_tokens=800, output_tokens=150)
    assert result.prompt_version == PROMPT_VERSION
    assert len(model.requests) == 1


def test_the_agent_sees_the_gaps_their_reasoning_and_the_whole_chart_but_not_met_criteria():
    _, model = run(writes(SCALE_REQUEST, THERAPY_REQUEST))
    sent = model.requests[0]
    message = sent.turns[0].text
    assert "demo.scale: A disability scale is recorded." in message
    assert "Evidence review: No disability scale recorded." in message
    assert "did some therapy last year" in message  # the note text itself
    assert "demo.duration" not in message
    assert sent.system == load_prompt(PROMPT_VERSION)
    [tool] = sent.tools
    assert tool.input_schema["properties"]["criterion_id"]["enum"] == ["demo.scale", "demo.therapy"]


@pytest.mark.parametrize(
    "bad,problem",
    [
        ({**SCALE_REQUEST, "criterion_id": "demo.duration"}, "is not one of the gaps"),
        ({**SCALE_REQUEST, "request": "Record ODI."}, "8 to 120 words"),
        ({**SCALE_REQUEST, "request": "word " * 121}, "8 to 120 words"),
        ({**SCALE_REQUEST, "note_type": " "}, "note_type"),
    ],
)
def test_a_bad_request_is_rejected_and_can_be_corrected(bad, problem):
    result, model = run(writes(bad, THERAPY_REQUEST), writes(SCALE_REQUEST))
    error = model.requests[1].turns[-1].tool_results[0]
    assert error.is_error and problem in error.content
    assert [r.criterion_id for r in result.requests] == ["demo.scale", "demo.therapy"]
    assert result.requests[0].request == SCALE_REQUEST["request"]


def test_each_gap_gets_only_one_request():
    result, _ = run(writes(SCALE_REQUEST, SCALE_REQUEST, THERAPY_REQUEST))
    results = result.turns[2].tool_results
    assert results[1].is_error and "already has a request" in results[1].content
    assert len(result.requests) == 2


def test_a_model_that_stops_early_is_reminded_then_gaps_left_are_reported():
    result, model = run(
        writes(SCALE_REQUEST),
        ModelResponse(text="Done.", stop_reason=StopReason.END_TURN),
        ModelResponse(stop_reason=StopReason.OTHER),
    )
    assert model.requests[2].turns[-1].text == "Write a request for each gap still open: demo.therapy."
    assert result.skipped == ["demo.therapy"] and result.stop is StopCause.MODEL_STOPPED


def test_prompt_is_generic():
    """Like the matcher's prompt: no numbers and nothing from a real policy."""
    prompt = load_prompt(PROMPT_VERSION)
    assert not re.search(r"\d", prompt)
    for policy in load_policies(Path("policies")).values():
        for criterion in policy.criteria:
            assert criterion.text not in prompt
            assert not any(quote.lower() in prompt.lower() for quote in criterion.source_quotes)
