"""The LLM judge: grades evidence relevance and documentation request quality, for evaluation only.

verify() proves each quote is in the chart; the judge asks whether it shows the criterion. It also checks
each documentation request. Its grades are never a merge gate: they are a measurement, trusted only as far
as their agreement with blind hand labels (see the calibration report). The judge is a different, stronger
model than the agent, to limit a model grading its own kind of work kindly.

For a request, the judge answers four yes/no checks and the overall grade is worked out in code, the same
way decide() works out the decision from the agent's findings.

    python -m evals.judge record      # needs ANTHROPIC_API_KEY; about 115 Opus calls
    python -m evals.judge replay
"""

import argparse
import threading
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Self

from pydantic import ValidationError, model_validator

from evals.gold import GoldCase
from evals.judge_items import GRADES, SAMPLE, ItemKind, JudgeItem, build_items, load_sample
from evals.runner import GOLD, POLICIES, RECORDINGS, Mode
from evals.scoring import PRICES
from priorauth.llm.base import ModelClient, ToolCall, ToolResult, ToolSpec, Turn, Usage
from priorauth.llm.replay import RecordingClient, ReplayClient, StaleRecording, load_recording, save_recording
from priorauth.matcher import tool_loop
from priorauth.policies import load_policies
from priorauth.schemas import Contract, NonBlank
from priorauth.tools import ToolError, describe, strict_object

JUDGE_MODEL = "claude-opus-5-5"
PROMPT_VERSION = "judge_v1"
PROMPT = Path(__file__).parent / "prompts" / f"{PROMPT_VERSION}.md"
MAX_STEPS = 3
WORKERS = 4  # items graded at once; each has its own conversation and recording
REQUEST_CHECKS = ["specific", "asks_only_what_is_needed", "invents_nothing", "says_where"]


class Grade(Contract):
    item_id: str
    grade: str
    reasoning: NonBlank
    checks: dict[str, bool] = {}  # request items: the four checks the grade is worked out from


class Judgement(Contract):
    item_id: str
    grade: Grade | None = None
    error: str = ""  # a missing or stale recording, or a judge that never graded
    usage: Usage = Usage()
    turns: list[Turn] = []

    @model_validator(mode="after")
    def _grade_or_error(self) -> Self:
        if (self.grade is None) == (not self.error):
            raise ValueError("a judgement has either a grade or an error")
        return self


def request_grade(checks: dict[str, bool]) -> str:
    """A request is acceptable only if every check passes."""
    return "acceptable" if all(checks[name] for name in REQUEST_CHECKS) else "needs_revision"


class GradeTool:
    """The one grading tool for one item. The model grades once; the grade is checked before it is kept."""

    def __init__(self, item: JudgeItem) -> None:
        self.item = item
        self.grade: Grade | None = None

    def remaining(self) -> list[str]:
        return [] if self.grade else [self.item.item_id]

    def specs(self) -> list[ToolSpec]:
        if self.item.kind is ItemKind.REQUEST:
            fields = {name: {"type": "boolean"} for name in REQUEST_CHECKS}
            return [
                ToolSpec(
                    name="grade_request",
                    description="Grade this request.",
                    input_schema=strict_object({**fields, "reasoning": {"type": "string"}}),
                )
            ]
        return [
            ToolSpec(
                name="grade_evidence",
                description="Grade this evidence.",
                input_schema=strict_object(
                    {"grade": {"type": "string", "enum": GRADES[ItemKind.EVIDENCE]}, "reasoning": {"type": "string"}}
                ),
            )
        ]

    def run(self, call: ToolCall) -> ToolResult:
        try:
            if call.name != self.specs()[0].name:
                raise ToolError(f"use {self.specs()[0].name} for this item")
            if self.grade:
                raise ToolError("this item is already graded")
            self.grade = self._grade(call.input)
            return ToolResult(tool_call_id=call.id, content="Recorded.")
        except ValidationError as error:
            message = describe(error)
        except ToolError as error:
            message = str(error)
        return ToolResult(tool_call_id=call.id, content=message, is_error=True)

    def _grade(self, args: dict[str, Any]) -> Grade:
        item_id = self.item.item_id
        if self.item.kind is ItemKind.REQUEST:
            checks = {name: args.get(name) for name in REQUEST_CHECKS}
            if not all(isinstance(value, bool) for value in checks.values()):
                raise ToolError(f"answer each of {', '.join(REQUEST_CHECKS)} with true or false")
            return Grade(
                item_id=item_id, grade=request_grade(checks), reasoning=args.get("reasoning", ""), checks=checks
            )
        if args.get("grade") not in GRADES[ItemKind.EVIDENCE]:
            raise ToolError(f"grade must be one of {GRADES[ItemKind.EVIDENCE]}")
        return Grade(item_id=item_id, grade=args["grade"], reasoning=args.get("reasoning", ""))


def item_message(item: JudgeItem, case: GoldCase) -> str:
    """What the judge sees: the criterion, the claim or request, and the notes. Never who made it or why."""

    def note_block(notes) -> str:
        return "\n".join(
            f'<note id="{n.note_id}" date="{n.date}" type="{n.note_type}">\n{n.text}\n</note>' for n in notes
        )

    if item.kind is ItemKind.REQUEST:
        return (
            f"<request_item>\n<criterion>{item.criterion_text}</criterion>\n"
            f'<request note_type="{item.note_type}">\n{item.request}\n</request>\n</request_item>\n\n'
            f"<notes>\n{note_block(case.request.notes)}\n</notes>"
        )
    cited = {q.note_id for q in item.quotes}
    quotes = "\n".join(f'- {q.note_id}: "{q.quote}"' for q in item.quotes)
    return (
        f"<evidence_item>\n<criterion>{item.criterion_text}</criterion>\n"
        f"<claim>The criterion is met, citing:\n{quotes}\n</claim>\n</evidence_item>\n\n"
        f"<notes>\n{note_block(n for n in case.request.notes if n.note_id in cited)}\n</notes>"
    )


def judge_item(client: ModelClient, item: JudgeItem, case: GoldCase, model: str = JUDGE_MODEL) -> Judgement:
    tool = GradeTool(item)
    loop = tool_loop(
        client,
        model=model,
        system=PROMPT.read_text(encoding="utf-8"),
        first_message=item_message(item, case),
        tools=tool,
        max_steps=MAX_STEPS,
        reminder="Grade the item with the tool.",
    )
    if tool.grade is None:
        return Judgement(item_id=item.item_id, error=f"no grade ({loop.stop})", usage=loop.usage, turns=loop.turns)
    return Judgement(item_id=item.item_id, grade=tool.grade, usage=loop.usage, turns=loop.turns)


def judged_items(items: list[JudgeItem], sample: list[str]) -> list[JudgeItem]:
    """Every evidence item and request, plus the planted items in the labelled sample."""
    chosen = set(sample)
    return [i for i in items if i.kind is not ItemKind.PLANTED or i.item_id in chosen]


def judge_path(recordings: Path, item_id: str) -> Path:
    return recordings / "judge" / f"{item_id}.json"


def run_judge(
    mode: Mode,
    items: list[JudgeItem],
    cases: dict[str, GoldCase],
    *,
    client: ModelClient | None = None,
    recordings: Path = RECORDINGS,
    model: str = JUDGE_MODEL,
    workers: int = WORKERS,
    on_result: Callable[[Judgement], None] | None = None,
) -> list[Judgement]:
    """One judgement per item, in item order. Missing or stale recordings are errors, not crashes."""
    if mode is Mode.RECORD and client is None:
        raise ValueError("record mode needs a live model client")
    lock = threading.Lock()

    def one(item: JudgeItem) -> Judgement:
        path = judge_path(recordings, item.item_id)
        try:
            if mode is Mode.RECORD:
                recorder = RecordingClient(client)
                result = judge_item(recorder, item, cases[item.case_id], model)
                save_recording(recorder.recording, path)
            else:
                if not path.exists():
                    raise StaleRecording(f"no recording at {path}")
                replay = ReplayClient(load_recording(path))
                result = judge_item(replay, item, cases[item.case_id], model)
                replay.check_finished()
        except StaleRecording as error:
            result = Judgement(item_id=item.item_id, error=str(error))
        if on_result:
            with lock:
                on_result(result)
        return result

    with ThreadPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(one, items))


def cost_usd(judgements: list[Judgement], model: str = JUDGE_MODEL) -> float | None:
    if model not in PRICES:
        return None
    per_input, per_output = PRICES[model]
    return sum(j.usage.input_tokens * per_input + j.usage.output_tokens * per_output for j in judgements) / 1_000_000


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the LLM judge over the judge items.")
    parser.add_argument("mode", choices=[mode.value for mode in Mode])
    parser.add_argument("--limit", type=int, help="grade only the first N items (to measure cost first)")
    parser.add_argument("--workers", type=int, default=WORKERS)
    parser.add_argument("--gold", type=Path, default=GOLD)
    parser.add_argument("--recordings", type=Path, default=RECORDINGS)
    parser.add_argument("--policies", type=Path, default=POLICIES)
    parser.add_argument("--sample", type=Path, default=SAMPLE)
    args = parser.parse_args(argv)
    client = None
    if args.mode == Mode.RECORD:
        from priorauth.llm.claude import ClaudeClient  # needs the agent extra and ANTHROPIC_API_KEY

        client = ClaudeClient()
    items, cases = build_items(load_policies(args.policies), args.gold, args.recordings)
    chosen = judged_items(items, load_sample(args.sample))[: args.limit]
    done = []

    def progress(judgement: Judgement) -> None:
        done.append(judgement)
        outcome = judgement.error or judgement.grade.grade
        print(f"{len(done)}/{len(chosen)} {judgement.item_id}: {outcome}", flush=True)

    results = run_judge(
        Mode(args.mode),
        chosen,
        cases,
        client=client,
        recordings=args.recordings,
        workers=args.workers,
        on_result=progress,
    )
    errors = [j for j in results if j.error]
    cost = cost_usd(results)
    print(f"{len(results)} items, {len(errors)} errors, cost {'unknown' if cost is None else f'${cost:.2f}'}")
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
