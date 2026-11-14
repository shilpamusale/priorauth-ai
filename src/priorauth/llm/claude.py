"""Claude adapter: translates the model interface to and from the Anthropic SDK.

This is the only module that imports the SDK. By default the SDK reads ANTHROPIC_API_KEY from the
environment; tests pass in a fake SDK client instead, so CI never needs a key.
"""

from typing import Any

import anthropic

from priorauth.llm.base import ModelRequest, ModelResponse, StopReason, ToolCall, Turn, Usage

STOP_REASONS = {"end_turn": StopReason.END_TURN, "tool_use": StopReason.TOOL_USE, "max_tokens": StopReason.MAX_TOKENS}


def to_message(turn: Turn) -> dict[str, Any]:
    """One turn as an Anthropic message. In a user message, tool results must come before any text."""
    blocks: list[dict[str, Any]] = [
        {"type": "tool_result", "tool_use_id": r.tool_call_id, "content": r.content, "is_error": r.is_error}
        for r in turn.tool_results
    ]
    if turn.text:
        blocks.append({"type": "text", "text": turn.text})
    blocks += [{"type": "tool_use", "id": c.id, "name": c.name, "input": c.input} for c in turn.tool_calls]
    return {"role": turn.role, "content": blocks}


def to_params(request: ModelRequest) -> dict[str, Any]:
    """The keyword arguments for messages.create."""
    params: dict[str, Any] = {
        "model": request.model,
        "system": request.system,
        "max_tokens": request.max_tokens,
        "messages": [to_message(turn) for turn in request.turns],
    }
    if request.tools:
        params["tools"] = [
            {"name": t.name, "description": t.description, "input_schema": t.input_schema, "strict": t.strict}
            for t in request.tools
        ]
    return params


def from_message(message: anthropic.types.Message) -> ModelResponse:
    """Claude's reply as a ModelResponse. Any block type the agent does not handle is an error, not skipped."""
    if unexpected := sorted({block.type for block in message.content} - {"text", "tool_use"}):
        raise ValueError(f"unsupported content blocks in Claude's reply: {unexpected}")
    return ModelResponse(
        text="\n".join(block.text for block in message.content if block.type == "text"),
        tool_calls=[
            ToolCall(id=block.id, name=block.name, input=block.input)
            for block in message.content
            if block.type == "tool_use"
        ],
        stop_reason=STOP_REASONS.get(message.stop_reason, StopReason.OTHER),
        usage=Usage(input_tokens=message.usage.input_tokens, output_tokens=message.usage.output_tokens),
    )


class ClaudeClient:
    """A ModelClient backed by Claude."""

    def __init__(self, sdk_client: anthropic.Anthropic | None = None) -> None:
        self._sdk = sdk_client or anthropic.Anthropic()

    def complete(self, request: ModelRequest) -> ModelResponse:
        return from_message(self._sdk.messages.create(**to_params(request)))
