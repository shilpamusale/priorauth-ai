"""Run the pipeline on a gold split, several trials per case, live or from recordings.

    record   call the live model and save each trial's exchanges under evals/recordings/
    replay   play the saved recordings back with no API key; this is what CI runs

Either pipeline can run: the single agent (recordings in evals/recordings/<split>/) or the multi-agent
orchestrator (recordings in evals/recordings/multi_agent/<split>/).

Each trial gives a TrialResult: its checks and audit record, or the error that stopped it. A missing or
stale recording is an error for that trial, not a crash, so one run reports every problem at once.

    python -m evals.runner record --cases dev_001 --trials 1    # needs ANTHROPIC_API_KEY
    python -m evals.runner replay --summary results.json
    python -m evals.runner replay --pipeline multi_agent
"""

import argparse
from collections.abc import Callable
from enum import StrEnum
from pathlib import Path
from typing import Self

from pydantic import model_validator

from evals.checks import CheckResult, run_checks
from evals.gold import SPLITS, GoldCase, load_split
from priorauth.llm.base import ModelClient
from priorauth.llm.replay import RecordingClient, ReplayClient, StaleRecording, load_recording, save_recording
from priorauth.matcher import DEFAULT_MODEL
from priorauth.pipeline import AuditRecord, Pipeline, run_case
from priorauth.policies import load_policies
from priorauth.schemas import Contract, Policy

GOLD = Path("evals/gold")
RECORDINGS = Path("evals/recordings")
POLICIES = Path("policies")
TRIALS = 3  # the agent is not deterministic, so each case runs more than once


class Mode(StrEnum):
    RECORD = "record"
    REPLAY = "replay"


class TrialResult(Contract):
    case_id: str
    trial: int
    checks: list[CheckResult] = []
    record: AuditRecord | None = None
    error: str = ""  # why the trial produced no record

    @model_validator(mode="after")
    def _record_or_error(self) -> Self:
        if (self.record is None) == (not self.error):
            raise ValueError("a trial has either a record or an error")
        return self


def recording_path(root: Path, split: str, case_id: str, trial: int) -> Path:
    return root / split / case_id / f"trial_{trial}.json"


def pipeline_root(recordings: Path, pipeline: Pipeline) -> Path:
    """Where a pipeline's recordings live. The single agent's stay where they have always been."""
    return recordings if pipeline is Pipeline.SINGLE_AGENT else recordings / pipeline.value


def record_trial(
    case: GoldCase,
    policy: Policy,
    client: ModelClient,
    path: Path,
    model: str,
    pipeline: Pipeline = Pipeline.SINGLE_AGENT,
) -> AuditRecord:
    recorder = RecordingClient(client)
    record = run_case(recorder, policy, case.request, model=model, pipeline=pipeline)
    save_recording(recorder.recording, path)
    return record


def replay_trial(
    case: GoldCase, policy: Policy, path: Path, model: str, pipeline: Pipeline = Pipeline.SINGLE_AGENT
) -> AuditRecord:
    if not path.exists():
        raise StaleRecording(f"no recording at {path}")
    replay = ReplayClient(load_recording(path))
    record = run_case(replay, policy, case.request, model=model, pipeline=pipeline)
    replay.check_finished()
    return record


def run_split(
    split: str,
    mode: Mode,
    policies: dict[str, Policy],
    *,
    client: ModelClient | None = None,
    gold: Path = GOLD,
    recordings: Path = RECORDINGS,
    trials: int = TRIALS,
    cases: list[str] | None = None,
    model: str = DEFAULT_MODEL,
    allow_heldout: bool = False,
    on_result: Callable[[TrialResult], None] | None = None,
    pipeline: Pipeline = Pipeline.SINGLE_AGENT,
) -> list[TrialResult]:
    """Every trial of every selected case, in case order. on_result sees each trial as soon as it ends."""
    if split == "heldout" and not allow_heldout:
        raise ValueError("held-out cases run only in the final evaluation; pass allow_heldout=True for that run")
    if mode is Mode.RECORD and client is None:
        raise ValueError("record mode needs a live model client")
    gold_cases = load_split(gold, split, policies)
    if unknown := sorted(set(cases or []) - gold_cases.keys()):
        raise ValueError(f"no such cases in {split}: {unknown}")
    results = []
    for case_id, case in gold_cases.items():
        if cases and case_id not in cases:
            continue
        policy = policies[case.policy_id]
        for trial in range(1, trials + 1):
            path = recording_path(pipeline_root(recordings, pipeline), split, case_id, trial)
            try:
                if mode is Mode.RECORD:
                    record = record_trial(case, policy, client, path, model, pipeline)
                else:
                    record = replay_trial(case, policy, path, model, pipeline)
            except StaleRecording as error:
                result = TrialResult(case_id=case_id, trial=trial, error=str(error))
            else:
                result = TrialResult(
                    case_id=case_id, trial=trial, checks=run_checks(case, policy, record), record=record
                )
            results.append(result)
            if on_result:
                on_result(result)
    return results


def summary_line(result: TrialResult) -> str:
    head = f"{result.case_id} trial {result.trial}:"
    if result.error:
        return f"{head} ERROR {result.error}"
    failed = [f"{c.name}{' (hard gate)' if c.hard_gate else ''}" for c in result.checks if not c.passed]
    outcome = f"failed {', '.join(failed)}" if failed else "all checks passed"
    return f"{head} {result.record.determination.decision}, {outcome}"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the pipeline on a gold split.")
    parser.add_argument("mode", choices=[mode.value for mode in Mode])
    parser.add_argument("--split", default="dev", choices=SPLITS)
    parser.add_argument("--cases", nargs="+", help="case IDs to run (default: the whole split)")
    parser.add_argument("--trials", type=int, default=TRIALS)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--pipeline", default=Pipeline.SINGLE_AGENT.value, choices=[p.value for p in Pipeline])
    parser.add_argument("--allow-heldout", action="store_true", help="only for the final evaluation")
    parser.add_argument("--gold", type=Path, default=GOLD)
    parser.add_argument("--recordings", type=Path, default=RECORDINGS)
    parser.add_argument("--policies", type=Path, default=POLICIES)
    parser.add_argument("--min-decision-accuracy", type=float, default=0.0, help="CI gate threshold")
    parser.add_argument("--summary", type=Path, help="also write the summary as JSON to this file")
    parser.add_argument("--track", action="store_true", help="log the run to MLflow (needs the track extra)")
    args = parser.parse_args(argv)
    client = None
    if args.mode == Mode.RECORD:
        from priorauth.llm.claude import ClaudeClient  # needs the agent extra and ANTHROPIC_API_KEY

        client = ClaudeClient()
    policies = load_policies(args.policies)
    results = run_split(
        args.split,
        Mode(args.mode),
        policies,
        client=client,
        gold=args.gold,
        recordings=args.recordings,
        trials=args.trials,
        cases=args.cases,
        model=args.model,
        allow_heldout=args.allow_heldout,
        on_result=lambda result: print(summary_line(result), flush=True),  # live progress during long runs
        pipeline=Pipeline(args.pipeline),
    )
    from evals.scoring import format_summary, gate, summarize  # scoring imports TrialResult from this module

    cases = load_split(args.gold, args.split, policies)
    summary = summarize(results, cases)
    print(format_summary(summary))
    if args.summary:
        args.summary.write_text(summary.model_dump_json(indent=1) + "\n", encoding="utf-8")
    if args.track:
        from evals.tracking import log_run  # tracking imports this module too

        print(f"MLflow run {log_run(summary, results, cases, policies, split=args.split, mode=args.mode)}")
    failures = gate(summary, args.min_decision_accuracy)
    print("gate failed: " + "; ".join(failures) if failures else "gate passed")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
