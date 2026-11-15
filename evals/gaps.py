"""Run the documentation agent on the dev cases that have documentation gaps.

Its input is the deployed single agent's first recorded trial of each case, replayed with no model call.
Its own model calls are recorded under evals/recordings/documentation/ and replayed in CI, so a change to
the single agent's recordings, the prompt or the chart makes these recordings stale too.

    python -m evals.gaps record      # needs ANTHROPIC_API_KEY
    python -m evals.gaps replay
"""

import argparse
from collections.abc import Callable
from pathlib import Path
from typing import Self

from pydantic import model_validator

from evals.gold import load_split
from evals.runner import GOLD, POLICIES, RECORDINGS, Mode, recording_path, replay_trial
from evals.scoring import PRICES
from priorauth.documentation import DocumentationRun, gaps, write_requests
from priorauth.llm.base import ModelClient
from priorauth.llm.replay import RecordingClient, ReplayClient, StaleRecording, load_recording, save_recording
from priorauth.matcher import DEFAULT_MODEL
from priorauth.policies import load_policies
from priorauth.schemas import Contract, Policy

SPLIT = "dev"  # never the held-out cases
TRIAL = 1  # the documentation agent works from the single agent's first trial


class GapResult(Contract):
    case_id: str
    run: DocumentationRun | None = None
    error: str = ""

    @model_validator(mode="after")
    def _run_or_error(self) -> Self:
        if (self.run is None) == (not self.error):
            raise ValueError("a gap result has either a run or an error")
        return self


def documentation_path(recordings: Path, case_id: str) -> Path:
    return recording_path(recordings / "documentation", SPLIT, case_id, TRIAL)


def run_cost(run: DocumentationRun) -> float | None:
    if run.model not in PRICES:
        return None
    per_input, per_output = PRICES[run.model]
    return (run.usage.input_tokens * per_input + run.usage.output_tokens * per_output) / 1_000_000


def run_gaps(
    mode: Mode,
    policies: dict[str, Policy],
    *,
    client: ModelClient | None = None,
    gold: Path = GOLD,
    recordings: Path = RECORDINGS,
    cases: list[str] | None = None,
    model: str = DEFAULT_MODEL,
    on_result: Callable[[GapResult], None] | None = None,
) -> list[GapResult]:
    """One result per dev case with gaps, in case order. Cases without gaps make no model call."""
    if mode is Mode.RECORD and client is None:
        raise ValueError("record mode needs a live model client")
    gold_cases = load_split(gold, SPLIT, policies)
    if unknown := sorted(set(cases or []) - gold_cases.keys()):
        raise ValueError(f"no such cases in {SPLIT}: {unknown}")
    results = []
    for case_id, case in gold_cases.items():
        if cases and case_id not in cases:
            continue
        policy = policies[case.policy_id]
        try:
            evidence = replay_trial(case, policy, recording_path(recordings, SPLIT, case_id, TRIAL), model)
            if not gaps(evidence.determination):
                continue
            path = documentation_path(recordings, case_id)
            if mode is Mode.RECORD:
                recorder = RecordingClient(client)
                run = write_requests(recorder, policy, case.request, evidence.determination, model=model)
                save_recording(recorder.recording, path)
            else:
                if not path.exists():
                    raise StaleRecording(f"no recording at {path}")
                replay = ReplayClient(load_recording(path))
                run = write_requests(replay, policy, case.request, evidence.determination, model=model)
                replay.check_finished()
        except StaleRecording as error:
            result = GapResult(case_id=case_id, error=str(error))
        else:
            result = GapResult(case_id=case_id, run=run)
        results.append(result)
        if on_result:
            on_result(result)
    return results


def report(result: GapResult) -> str:
    """The requests a case got, one per line, for reading after a run."""
    if result.error:
        return f"{result.case_id}: ERROR {result.error}"
    run = result.run
    lines = [f"{result.case_id}: {len(run.requests)} request{'s' * (len(run.requests) != 1)}"]
    for request in run.requests:
        name = request.criterion_id.split(".", 1)[-1]
        lines.append(f"  - {name} ({request.note_type}): {request.request}")
    if run.skipped:
        lines.append(f"  gaps left without a request: {', '.join(run.skipped)}")
    return "\n".join(lines)


def failures(results: list[GapResult]) -> list[str]:
    """Why the gate fails: a case that could not run, or a gap left without a request."""
    return [f"{r.case_id}: {r.error}" for r in results if r.error] + [
        f"{r.case_id}: no request for {', '.join(r.run.skipped)}" for r in results if r.run and r.run.skipped
    ]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the documentation agent on the dev cases with gaps.")
    parser.add_argument("mode", choices=[mode.value for mode in Mode])
    parser.add_argument("--cases", nargs="+", help="case IDs to run (default: every dev case with gaps)")
    parser.add_argument("--gold", type=Path, default=GOLD)
    parser.add_argument("--recordings", type=Path, default=RECORDINGS)
    parser.add_argument("--policies", type=Path, default=POLICIES)
    args = parser.parse_args(argv)
    client = None
    if args.mode == Mode.RECORD:
        from priorauth.llm.claude import ClaudeClient  # needs the agent extra and ANTHROPIC_API_KEY

        client = ClaudeClient()
    results = run_gaps(
        Mode(args.mode),
        load_policies(args.policies),
        client=client,
        gold=args.gold,
        recordings=args.recordings,
        cases=args.cases,
        on_result=lambda result: print(report(result), flush=True),
    )
    runs = [r.run for r in results if r.run]
    costs = [run_cost(run) for run in runs]
    cost = "unknown" if None in costs else f"${sum(costs):.2f}"
    requests = sum(len(run.requests) for run in runs)
    print(f"{len(results)} cases with gaps, {requests} requests, {len(results) - len(runs)} errors, cost {cost}")
    problems = failures(results)
    print("gate failed: " + "; ".join(problems) if problems else "gate passed")
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
