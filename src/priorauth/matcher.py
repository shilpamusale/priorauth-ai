"""The single-agent matcher: the model reads the chart with tools and submits one finding per criterion.

The loop is bounded. It ends when every criterion has a finding, when the step limit is reached, or when
the model stops in a way the loop cannot continue from. Criteria still without a finding are reported as
skipped, and decide() escalates a case whose required criteria were not all checked.
"""

from enum import StrEnum
from importlib import resources

from priorauth.llm.base import ModelClient, ModelRequest, StopReason, Turn, Usage
from priorauth.schemas import Contract, CriterionResult, PARequest, Policy
from priorauth.tools import ChartTools

DEFAULT_MODEL = "claude-sonnet-5-5"
PROMPT_VERSION = "matcher_v1"
MAX_STEPS = 20  # model calls per case; bounds the cost of a model that never finishes


class StopCause(StrEnum):
    COMPLETE = "complete"  # every criterion has a finding
    STEP_LIMIT = "step_limit"
    CUT_OFF = "cut_off"  # a reply hit max_tokens; its tool calls may be incomplete, so none are run
    MODEL_STOPPED = "model_stopped"  # the model stopped for another reason, such as a refusal


class MatchRun(Contract):
    """Everything the matcher did on one case, for the pipeline and the audit record."""

    model: str
    prompt_version: str
    findings: list[CriterionResult]  # in policy order
    skipped: list[str]  # criteria with no finding when the loop ended
    stop: StopCause
    steps: int  # model calls made
    usage: Usage
    turns: list[Turn]  # the whole conversation, including every tool call and result


def load_prompt(version: str = PROMPT_VERSION) -> str:
    return (resources.files("priorauth") / "prompts" / f"{version}.md").read_text(encoding="utf-8")


def case_message(policy: Policy, request: PARequest, tools: ChartTools) -> str:
    """The first user message: the request, the criteria, and the note index. Note text is read with tools."""
    criteria = "\n".join(f"- {c.criterion_id}: {c.text}" for c in policy.criteria)
    return (
        f"<request>\nService code: {request.service_code}\nDiagnosis codes: {', '.join(request.icd10_codes)}\n"
        f"Request date: {request.request_date}\n</request>\n\n"
        f"<policy>\n{policy.title}\n\nCriteria:\n{criteria}\n</policy>\n\n"
        f"<notes>\n{tools.note_index()}\n</notes>"
    )


def match(
    client: ModelClient, policy: Policy, request: PARequest, model: str = DEFAULT_MODEL, max_steps: int = MAX_STEPS
) -> MatchRun:
    """Run the tool loop on one case."""
    tools = ChartTools(policy, request)
    system = load_prompt()
    turns = [Turn(role="user", text=case_message(policy, request, tools))]
    usage = Usage()
    steps, stop = 0, StopCause.STEP_LIMIT
    while steps < max_steps:
        steps += 1
        response = client.complete(ModelRequest(model=model, system=system, turns=turns, tools=tools.specs()))
        usage = Usage(
            input_tokens=usage.input_tokens + response.usage.input_tokens,
            output_tokens=usage.output_tokens + response.usage.output_tokens,
        )
        if response.text or response.tool_calls:
            turns.append(response.as_turn())
        if response.stop_reason is StopReason.MAX_TOKENS:
            stop = StopCause.CUT_OFF
            break
        if response.tool_calls:
            turns.append(Turn(role="user", tool_results=[tools.run(call) for call in response.tool_calls]))
        elif response.stop_reason is StopReason.END_TURN and response.text:
            reminder = f"Submit a finding for each criterion still open: {', '.join(tools.remaining())}."
            turns.append(Turn(role="user", text=reminder))
        else:
            stop = StopCause.MODEL_STOPPED
            break
        if not tools.remaining():
            stop = StopCause.COMPLETE
            break
    return MatchRun(
        model=model,
        prompt_version=PROMPT_VERSION,
        findings=[tools.findings[c.criterion_id] for c in policy.criteria if c.criterion_id in tools.findings],
        skipped=tools.remaining(),
        stop=stop,
        steps=steps,
        usage=usage,
        turns=turns,
    )