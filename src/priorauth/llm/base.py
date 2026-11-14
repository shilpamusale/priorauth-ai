"""The model interface the agent depends on.

The matcher talks only to these types, never to a provider's SDK. Tests can script the model, recorded
runs can be replayed in CI, and another provider is one adapter file.
"""

from enum import StrEnum
from itertools import pairwise
from typing import Any, Literal, Protocol, Self, runtime_checkable

from pydantic import Field, model_validator

from priorauth.schemas import Contract


class ToolSpec(Contract):
    """A tool the model may call, described by a JSON Schema for its input."""

    name: str
    description: str
    input_schema: dict[str, Any]
    strict: bool = True  # the provider constrains the model's tool input to match the schema


class ToolCall(Contract):
    id: str
    name: str
    input: dict[str, Any]


class ToolResult(Contract):
    tool_call_id: str
    content: str
    is_error: bool = False  # tells the model its call failed, e.g. a quote not found in the note


class Turn(Contract):
    """One message: the user side sends text or tool results, the assistant sends text or tool calls."""

    role: Literal["user", "assistant"]
    text: str = ""
    tool_calls: list[ToolCall] = Field(default_factory=list)
    tool_results: list[ToolResult] = Field(default_factory=list)

    @model_validator(mode="after")
    def _parts_match_the_role(self) -> Self:
        if self.role == "user" and self.tool_calls:
            raise ValueError("only the assistant makes tool calls")
        if self.role == "assistant" and self.tool_results:
            raise ValueError("only the user side returns tool results")
        if not (self.text or self.tool_calls or self.tool_results):
            raise ValueError("a turn must carry text, tool calls or tool results")
        return self


class ModelRequest(Contract):
    """Everything sent to the model for one step of the conversation."""

    model: str
    system: str
    turns: list[Turn] = Field(min_length=1)
    tools: list[ToolSpec] = Field(default_factory=list)
    max_tokens: int = Field(default=4096, gt=0)

    @model_validator(mode="after")
    def _well_formed_conversation(self) -> Self:
        roles = [turn.role for turn in self.turns]
        if roles[0] != "user" or roles[-1] != "user" or any(a == b for a, b in pairwise(roles)):
            raise ValueError("turns must alternate user and assistant, starting and ending with the user")
        if self.turns[0].tool_results:
            raise ValueError("the first turn cannot answer tool calls")
        for previous, turn in pairwise(self.turns):
            if previous.role != "assistant":
                continue
            called = sorted(call.id for call in previous.tool_calls)
            answered = sorted(result.tool_call_id for result in turn.tool_results)
            if called != answered:
                raise ValueError("each tool call must be answered by exactly one tool result in the next turn")
        return self


class StopReason(StrEnum):
    END_TURN = "end_turn"  # the model finished its reply
    TOOL_USE = "tool_use"  # the model is waiting for tool results
    MAX_TOKENS = "max_tokens"  # the reply was cut off by the token limit
    OTHER = "other"


class Usage(Contract):
    input_tokens: int = Field(default=0, ge=0)
    output_tokens: int = Field(default=0, ge=0)


class ModelResponse(Contract):
    """The model's reply to one request."""

    text: str = ""
    tool_calls: list[ToolCall] = Field(default_factory=list)
    stop_reason: StopReason
    usage: Usage = Field(default_factory=Usage)

    def as_turn(self) -> Turn:
        """The reply as the assistant's turn, ready to append to the conversation."""
        return Turn(role="assistant", text=self.text, tool_calls=self.tool_calls)


@runtime_checkable
class ModelClient(Protocol):
    """Anything that answers a ModelRequest: the Claude adapter, a scripted fake or a replay of recordings."""

    def complete(self, request: ModelRequest) -> ModelResponse: ...