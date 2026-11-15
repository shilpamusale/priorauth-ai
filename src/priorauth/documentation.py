"""The documentation agent: for each criterion the chart does not document, a request to the clinic.

It runs after the evidence step, on the criteria found to have insufficient evidence. It sees those
criteria, the evidence step's reasoning for each, and the chart, and it writes one request per gap with
the write_request tool. It never changes a finding or the decision; its requests make a "needs
documentation" result something staff can act on.
"""

from typing import Any

from pydantic import Field, ValidationError

from priorauth.llm.base import ModelClient, ToolCall, ToolResult, ToolSpec, Turn, Usage
from priorauth.matcher import DEFAULT_MODEL, StopCause, load_prompt, tool_loop
from priorauth.schemas import Contract, CriterionResult, CriterionStatus, Determination, NonBlank, PARequest, Policy
from priorauth.tools import ToolError, describe, strict_object

PROMPT_VERSION = "documentation_v1"
MAX_STEPS = 4  # usually one call writes every request
MIN_WORDS, MAX_WORDS = 8, 120  # long enough to be specific, short enough to read at a glance


class DocumentationRequest(Contract):
    criterion_id: str
    request: NonBlank
    note_type: NonBlank


class DocumentationRun(Contract):
    model: str
    prompt_version: str
    requests: list[DocumentationRequest]  # in policy order
    skipped: list[str] = Field(default_factory=list)  # gaps with no request when the loop ended
    stop: StopCause
    steps: int
    usage: Usage
    turns: list[Turn]


def gaps(determination: Determination) -> list[CriterionResult]:
    """The criteria the chart does not document enough, in policy order."""
    return [r for r in determination.criteria if r.status is CriterionStatus.INSUFFICIENT_EVIDENCE]


class RequestTools:
    """The write_request tool for one case, and the requests written so far."""

    def __init__(self, gap_results: list[CriterionResult]) -> None:
        self.gaps = {result.criterion_id: result for result in gap_results}
        self.requests: dict[str, DocumentationRequest] = {}

    def remaining(self) -> list[str]:
        return [criterion_id for criterion_id in self.gaps if criterion_id not in self.requests]

    def specs(self) -> list[ToolSpec]:
        return [
            ToolSpec(
                name="write_request",
                description=(
                    "Write the documentation request for one gap. Each gap gets exactly one request, "
                    "which cannot be changed once written."
                ),
                input_schema=strict_object(
                    {
                        "criterion_id": {"type": "string", "enum": list(self.gaps)},
                        "request": {"type": "string"},
                        "note_type": {"type": "string"},
                    }
                ),
            )
        ]

    def run(self, call: ToolCall) -> ToolResult:
        try:
            if call.name != "write_request":
                raise ToolError(f"unknown tool {call.name!r}")
            return ToolResult(tool_call_id=call.id, content=self._write(call.input))
        except ValidationError as error:
            message = describe(error)
        except ToolError as error:
            message = str(error)
        return ToolResult(tool_call_id=call.id, content=message, is_error=True)

    def _write(self, args: dict[str, Any]) -> str:
        written = DocumentationRequest.model_validate(args)
        if written.criterion_id not in self.gaps:
            raise ToolError(f"{written.criterion_id!r} is not one of the gaps")
        if written.criterion_id in self.requests:
            raise ToolError(f"{written.criterion_id} already has a request")
        if not MIN_WORDS <= len(written.request.split()) <= MAX_WORDS:
            raise ToolError(f"a request must be {MIN_WORDS} to {MAX_WORDS} words long")
        self.requests[written.criterion_id] = written
        left = self.remaining()
        return f"Recorded the request for {written.criterion_id}. " + (
            f"Still to write: {', '.join(left)}." if left else "Every gap has a request."
        )


def gaps_message(policy: Policy, request: PARequest, gap_results: list[CriterionResult]) -> str:
    """The first user message: the request, each gap with the evidence step's reasoning, and the full chart."""
    texts = {c.criterion_id: c.text for c in policy.criteria}
    listed = "\n".join(
        f"- {r.criterion_id}: {texts[r.criterion_id]}\n  Evidence review: {r.reasoning or 'no reasoning given'}"
        for r in gap_results
    )
    notes = "\n".join(
        f'<note id="{n.note_id}" date="{n.date}" type="{n.note_type}">\n{n.text}\n</note>' for n in request.notes
    )
    return (
        f"<request>\nService code: {request.service_code}\nDiagnosis codes: {', '.join(request.icd10_codes)}\n"
        f"Request date: {request.request_date}\n</request>\n\n"
        f"<gaps>\n{listed}\n</gaps>\n\n<notes>\n{notes}\n</notes>"
    )


def write_requests(
    client: ModelClient,
    policy: Policy,
    request: PARequest,
    determination: Determination,
    model: str = DEFAULT_MODEL,
    max_steps: int = MAX_STEPS,
) -> DocumentationRun | None:
    """One request per documentation gap, or None when there are no gaps (and so no model call)."""
    gap_results = gaps(determination)
    if not gap_results:
        return None
    tools = RequestTools(gap_results)
    loop = tool_loop(
        client,
        model=model,
        system=load_prompt(PROMPT_VERSION),
        first_message=gaps_message(policy, request, gap_results),
        tools=tools,
        max_steps=max_steps,
        reminder="Write a request for each gap still open: {open}.",
    )
    return DocumentationRun(
        model=model,
        prompt_version=PROMPT_VERSION,
        requests=[tools.requests[r.criterion_id] for r in gap_results if r.criterion_id in tools.requests],
        skipped=tools.remaining(),
        stop=loop.stop,
        steps=loop.steps,
        usage=loop.usage,
        turns=loop.turns,
    )
