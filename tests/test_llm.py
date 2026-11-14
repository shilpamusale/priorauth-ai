"""The model interface rejects malformed conversations before anything is sent to a provider."""

import pytest
from pydantic import ValidationError

from priorauth.llm.base import (
    ModelClient,
    ModelRequest,
    ModelResponse,
    StopReason,
    ToolCall,
    ToolResult,
    ToolSpec,
    Turn,
)
from priorauth.llm.scripted import ScriptedClient

CALL = ToolCall(id="t1", name="read_note", input={"note_id": "n1"})
ASK = Turn(role="user", text="Review the chart.")
CALLS = Turn(role="assistant", tool_calls=[CALL])
ANSWER = Turn(role="user", tool_results=[ToolResult(tool_call_id="t1", content="note text")])


def request(*turns):
    return ModelRequest(model="m", system="s", turns=list(turns))


def test_tool_spec_is_strict_by_default():
    assert ToolSpec(name="read_note", description="d", input_schema={"type": "object"}).strict is True


def test_user_turn_cannot_make_tool_calls():
    with pytest.raises(ValidationError, match="only the assistant"):
        Turn(role="user", tool_calls=[CALL])


def test_assistant_turn_cannot_return_tool_results():
    with pytest.raises(ValidationError, match="only the user side"):
        Turn(role="assistant", tool_results=[ToolResult(tool_call_id="t1", content="x")])


def test_empty_turn_is_rejected():
    with pytest.raises(ValidationError, match="must carry"):
        Turn(role="user")


def test_valid_tool_round_trip_is_accepted():
    assert len(request(ASK, CALLS, ANSWER).turns) == 3


@pytest.mark.parametrize(
    "turns",
    [
        [CALLS, ANSWER],  # starts with the assistant
        [ASK, Turn(role="user", text="again")],  # two user turns in a row
        [ASK, CALLS],  # ends with the assistant
    ],
)
def test_turns_must_alternate_and_end_with_the_user(turns):
    with pytest.raises(ValidationError, match="alternate"):
        request(*turns)


def test_first_turn_cannot_answer_tool_calls():
    with pytest.raises(ValidationError, match="first turn"):
        request(ANSWER)


@pytest.mark.parametrize(
    "results",
    [
        [],  # the call is never answered
        [ToolResult(tool_call_id="t9", content="x")],  # answers a call that was not made
        [ToolResult(tool_call_id="t1", content="x"), ToolResult(tool_call_id="t1", content="y")],  # answered twice
    ],
)
def test_every_tool_call_needs_exactly_one_result(results):
    answer = Turn(role="user", text="ok", tool_results=results)
    with pytest.raises(ValidationError, match="exactly one tool result"):
        request(ASK, CALLS, answer)


def test_response_becomes_the_assistant_turn():
    response = ModelResponse(text="Checking.", tool_calls=[CALL], stop_reason=StopReason.TOOL_USE)
    assert response.as_turn() == Turn(role="assistant", text="Checking.", tool_calls=[CALL])


def test_request_survives_a_json_round_trip():
    """Recorded runs are stored as JSON, so a request must come back exactly as it was."""
    original = request(ASK, CALLS, ANSWER)
    assert ModelRequest.model_validate_json(original.model_dump_json()) == original


def test_scripted_client_plays_responses_in_order_and_records_requests():
    first = ModelResponse(tool_calls=[CALL], stop_reason=StopReason.TOOL_USE)
    second = ModelResponse(text="Done.", stop_reason=StopReason.END_TURN)
    client = ScriptedClient([first, second])
    assert client.complete(request(ASK)) == first
    assert client.complete(request(ASK, CALLS, ANSWER)) == second
    assert [len(r.turns) for r in client.requests] == [1, 3]


def test_scripted_client_keeps_a_snapshot_of_each_request():
    client = ScriptedClient([ModelResponse(text="ok", stop_reason=StopReason.END_TURN)])
    sent = request(ASK)
    client.complete(sent)
    sent.turns.append(CALLS)
    assert len(client.requests[0].turns) == 1


def test_scripted_client_fails_loudly_when_out_of_responses():
    with pytest.raises(RuntimeError, match="no responses left"):
        ScriptedClient([]).complete(request(ASK))


def test_scripted_client_satisfies_the_interface():
    assert isinstance(ScriptedClient([]), ModelClient)