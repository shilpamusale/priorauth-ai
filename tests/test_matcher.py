"""The matcher's tool loop, driven by a scripted model: no API key, no network."""

import datetime
import re
from pathlib import Path

import pytest

from priorauth.decision import decide
from priorauth.llm.base import ModelResponse, StopReason, ToolCall, Usage
from priorauth.llm.scripted import ScriptedClient
from priorauth.matcher import DEFAULT_MODEL, PROMPT_VERSION, StopCause, load_prompt, match
from priorauth.policies import load_policies
from priorauth.schemas import ClinicalNote, CriterionStatus, Decision, PARequest, Policy, PolicyCriterion

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
    icd10_codes=["M54.50", "M47.816"],
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
DURATION = {
    "criterion_id": "demo.duration",
    "status": "met",
    "evidence": [{"note_id": "n1", "quote": "Low back pain since March 2026"}],
    "reasoning": "Pain since March, over 6 months before the request.",
}
SCALE = {
    "criterion_id": "demo.scale",
    "status": "met",
    "evidence": [{"note_id": "n2", "quote": "ODI 44% at discharge"}],
    "reasoning": "ODI recorded.",
}


def reply(*calls, text="", stop=StopReason.TOOL_USE, tokens=(100, 20)):
    """One model reply; each call is (tool name, input)."""
    tool_calls = [ToolCall(id=f"toolu_{i}", name=name, input=args) for i, (name, args) in enumerate(calls)]
    return ModelResponse(
        text=text, tool_calls=tool_calls, stop_reason=stop, usage=Usage(input_tokens=tokens[0], output_tokens=tokens[1])
    )


def run(*replies, max_steps=10):
    client = ScriptedClient(list(replies))
    return match(client, POLICY, REQUEST, max_steps=max_steps), client


def test_complete_run_returns_findings_in_policy_order_and_stops_at_once():
    result, client = run(
        reply(("read_note", {"note_id": "n2"}), ("read_note", {"note_id": "n1"})),
        reply(("submit_finding", SCALE), ("submit_finding", DURATION)),
    )
    assert result.stop is StopCause.COMPLETE
    assert [f.criterion_id for f in result.findings] == ["demo.duration", "demo.scale"]
    assert result.skipped == []
    assert result.steps == 2 and len(client.requests) == 2  # no extra model call once every finding is in
    assert result.usage == Usage(input_tokens=200, output_tokens=40)
    assert (result.model, result.prompt_version) == (DEFAULT_MODEL, PROMPT_VERSION)
    assert decide(POLICY, result.findings) is Decision.READY_TO_SUBMIT


def test_first_request_carries_the_case_but_not_the_note_text():
    _, client = run(reply(("submit_finding", DURATION), ("submit_finding", SCALE)))
    request = client.requests[0]
    assert request.model == DEFAULT_MODEL
    assert request.system == load_prompt()
    assert [t.name for t in request.tools] == ["read_note", "search_chart", "submit_finding"]
    message = request.turns[0].text
    for expected in ["64493", "M54.50, M47.816", "2026-10-01", "Demo policy", "demo.duration: Pain has lasted"]:
        assert expected in message
    assert "n2 | 2026-09-01 | physical therapy discharge summary" in message
    assert "ODI 44%" not in message


def test_tool_results_go_back_to_the_model():
    _, client = run(
        reply(("read_note", {"note_id": "n2"})),
        reply(("submit_finding", DURATION), ("submit_finding", SCALE)),
    )
    results = client.requests[1].turns[-1].tool_results
    assert [r.tool_call_id for r in results] == ["toolu_0"]
    assert "ODI 44% at discharge" in results[0].content


def test_a_rejected_quote_can_be_corrected():
    wrong = {**SCALE, "evidence": [{"note_id": "n2", "quote": "ODI 48% at discharge"}]}
    result, client = run(
        reply(("submit_finding", DURATION), ("submit_finding", wrong)),
        reply(("submit_finding", SCALE)),
    )
    error = client.requests[1].turns[-1].tool_results[1]
    assert error.is_error and "is not in n2" in error.content
    assert result.stop is StopCause.COMPLETE
    assert result.findings[1].evidence[0].quote == "ODI 44% at discharge"


def test_step_limit_stops_the_loop_and_skipped_criteria_escalate():
    result, client = run(*[reply(("search_chart", {"query": "pain"})) for _ in range(5)], max_steps=3)
    assert result.stop is StopCause.STEP_LIMIT
    assert result.steps == 3 and len(client.requests) == 3
    assert result.findings == [] and result.skipped == ["demo.duration", "demo.scale"]
    assert decide(POLICY, result.findings) is Decision.ESCALATE


def test_model_that_stops_early_is_told_what_is_still_open():
    result, client = run(
        reply(("submit_finding", DURATION)),
        reply(text="I have reviewed the chart.", stop=StopReason.END_TURN),
        reply(("submit_finding", SCALE)),
    )
    assert client.requests[2].turns[-1].text == "Submit a finding for each criterion still open: demo.scale."
    assert result.stop is StopCause.COMPLETE


def test_a_cut_off_reply_is_not_acted_on():
    result, _ = run(reply(("submit_finding", DURATION), stop=StopReason.MAX_TOKENS))
    assert result.stop is StopCause.CUT_OFF
    assert result.findings == []


def test_an_empty_or_refused_reply_ends_the_run():
    result, _ = run(
        reply(("submit_finding", DURATION)),
        reply(stop=StopReason.OTHER),
    )
    assert result.stop is StopCause.MODEL_STOPPED
    assert [f.criterion_id for f in result.findings] == ["demo.duration"]
    assert result.skipped == ["demo.scale"]


def test_the_transcript_keeps_every_turn():
    result, _ = run(
        reply(("read_note", {"note_id": "n1"}), text="Reading the first note."),
        reply(("submit_finding", DURATION), ("submit_finding", SCALE)),
    )
    assert [t.role for t in result.turns] == ["user", "assistant", "user", "assistant", "user"]
    assert result.turns[1].text == "Reading the first note."
    assert result.findings[0].status is CriterionStatus.MET


def test_prompt_is_generic():
    """No numbers, so no thresholds, and nothing from a real policy: the policy arrives as data."""
    prompt = load_prompt()
    assert not re.search(r"\d", prompt)
    for policy in load_policies(Path("policies")).values():
        assert policy.policy_id not in prompt
        for criterion in policy.criteria:
            assert criterion.text not in prompt
            assert not any(quote.lower() in prompt.lower() for quote in criterion.source_quotes)


@pytest.mark.parametrize("version", ["matcher_v1"])
def test_prompt_versions_load(version):
    assert load_prompt(version).startswith("You check a patient's chart")