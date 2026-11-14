"""The Claude adapter speaks the Anthropic API's format, tested against a fake SDK client with no key."""

import typing

import anthropic
import pytest
from anthropic.types.message_create_params import MessageCreateParamsNonStreaming

from priorauth.llm.base import (
    ModelClient,
    ModelRequest,
    StopReason,
    ToolCall,
    ToolResult,
    ToolSpec,
    Turn,
    Usage,
)
from priorauth.llm.claude import ClaudeClient, to_params

TOOL = ToolSpec(name="read_note", description="Read one note.", input_schema={"type": "object"})
CALL = ToolCall(id="toolu_1", name="read_note", input={"note_id": "n1"})
REQUEST = ModelRequest(
    model="claude-sonnet-5-5",
    system="You review prior authorization charts.",
    tools=[TOOL],
    turns=[
        Turn(role="user", text="Review the chart."),
        Turn(role="assistant", text="Reading the note.", tool_calls=[CALL]),
        Turn(role="user", text="Continue.", tool_results=[ToolResult(tool_call_id="toolu_1", content="NRS 7/10")]),
    ],
)


def message(content, stop_reason="tool_use", input_tokens=120, output_tokens=30):
    """A reply built with the SDK's own Message type, so attribute names match the real SDK."""
    return anthropic.types.Message.model_validate(
        {
            "id": "msg_1",
            "type": "message",
            "role": "assistant",
            "model": "claude-sonnet-5-5",
            "content": content,
            "stop_reason": stop_reason,
            "stop_sequence": None,
            "usage": {"input_tokens": input_tokens, "output_tokens": output_tokens},
        }
    )


class FakeSDK:
    """Stands in for anthropic.Anthropic: records what messages.create receives and returns a set reply."""

    def __init__(self, reply):
        self.sent = []
        self.messages = self
        self._reply = reply

    def create(self, **params):
        self.sent.append(params)
        return self._reply


def test_request_is_sent_in_the_anthropic_format():
    assert to_params(REQUEST) == {
        "model": "claude-sonnet-5-5",
        "system": "You review prior authorization charts.",
        "max_tokens": 4096,
        "messages": [
            {"role": "user", "content": [{"type": "text", "text": "Review the chart."}]},
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": "Reading the note."},
                    {"type": "tool_use", "id": "toolu_1", "name": "read_note", "input": {"note_id": "n1"}},
                ],
            },
            {
                "role": "user",
                "content": [
                    {"type": "tool_result", "tool_use_id": "toolu_1", "content": "NRS 7/10", "is_error": False},
                    {"type": "text", "text": "Continue."},
                ],
            },
        ],
        "tools": [
            {"name": "read_note", "description": "Read one note.", "input_schema": {"type": "object"}, "strict": True}
        ],
    }


def test_tools_are_left_out_when_there_are_none():
    assert "tools" not in to_params(REQUEST.model_copy(update={"tools": []}))


def test_every_field_sent_is_one_the_sdk_defines():
    """Catches a misspelled key, which the SDK would otherwise pass along without complaint."""
    allowed = {
        "tool_result": anthropic.types.ToolResultBlockParam,
        "text": anthropic.types.TextBlockParam,
        "tool_use": anthropic.types.ToolUseBlockParam,
    }
    params = to_params(REQUEST)
    for block in (block for msg in params["messages"] for block in msg["content"]):
        assert set(block) <= set(typing.get_type_hints(allowed[block["type"]])), block
    for tool in params["tools"]:
        assert set(tool) <= set(typing.get_type_hints(anthropic.types.ToolParam)), tool
    assert set(params) <= set(typing.get_type_hints(MessageCreateParamsNonStreaming))


def test_reply_text_tool_calls_and_usage_are_read():
    reply = message(
        [
            {"type": "text", "text": "Checking the PT note."},
            {"type": "tool_use", "id": "toolu_2", "name": "read_note", "input": {"note_id": "n2"}},
        ]
    )
    sdk = FakeSDK(reply)
    response = ClaudeClient(sdk).complete(REQUEST)
    assert sdk.sent == [to_params(REQUEST)]
    assert response.text == "Checking the PT note."
    assert response.tool_calls == [ToolCall(id="toolu_2", name="read_note", input={"note_id": "n2"})]
    assert response.stop_reason is StopReason.TOOL_USE
    assert response.usage == Usage(input_tokens=120, output_tokens=30)


@pytest.mark.parametrize(
    "claude_reason,ours",
    [
        ("end_turn", StopReason.END_TURN),
        ("tool_use", StopReason.TOOL_USE),
        ("max_tokens", StopReason.MAX_TOKENS),
        ("refusal", StopReason.OTHER),
        ("stop_sequence", StopReason.OTHER),
    ],
)
def test_stop_reasons_are_mapped(claude_reason, ours):
    reply = message([{"type": "text", "text": "Done."}], stop_reason=claude_reason)
    assert ClaudeClient(FakeSDK(reply)).complete(REQUEST).stop_reason is ours


def test_unsupported_block_fails_loudly():
    reply = message([{"type": "thinking", "thinking": "Let me think.", "signature": "sig"}], stop_reason="end_turn")
    with pytest.raises(ValueError, match="unsupported content blocks"):
        ClaudeClient(FakeSDK(reply)).complete(REQUEST)


def test_reply_can_continue_the_conversation():
    """The reply's assistant turn plus tool results makes a valid next request."""
    reply = message([{"type": "tool_use", "id": "toolu_2", "name": "read_note", "input": {"note_id": "n2"}}])
    response = ClaudeClient(FakeSDK(reply)).complete(REQUEST)
    answer = Turn(role="user", tool_results=[ToolResult(tool_call_id="toolu_2", content="ODI 44%")])
    follow_up = REQUEST.model_copy(update={"turns": [*REQUEST.turns, response.as_turn(), answer]})
    assert ModelRequest.model_validate(follow_up.model_dump())


def test_claude_client_satisfies_the_interface():
    assert isinstance(ClaudeClient(FakeSDK(None)), ModelClient)
