"""Data for the demo: recorded trials, replayed through the real pipeline.

The demo never calls a model. Each page replays a recording of a live run through the same match, verify
and decide code that CI tests, so what it shows is what the code does with those model replies. Only dev
cases appear; held-out cases stay unseen until the final evaluation.
"""

import html
import os
from pathlib import Path

from evals.checks import run_checks
from evals.gold import GoldCase
from evals.runner import GOLD, POLICIES, RECORDINGS, TrialResult, recording_path, replay_trial
from priorauth.ingest import normalize
from priorauth.llm.replay import StaleRecording
from priorauth.matcher import DEFAULT_MODEL
from priorauth.schemas import Policy

SPLIT = "dev"


def folders() -> tuple[Path, Path, Path]:
    """Gold cases, recordings and policies. Environment variables override the defaults (used by tests)."""
    return (
        Path(os.environ.get("PRIORAUTH_GOLD", GOLD)),
        Path(os.environ.get("PRIORAUTH_RECORDINGS", RECORDINGS)),
        Path(os.environ.get("PRIORAUTH_POLICIES", POLICIES)),
    )


def recorded_trials(recordings: Path, case_ids: list[str]) -> dict[str, list[int]]:
    """Trial numbers that have a recording, for each case with at least one."""
    found = {}
    for case_id in case_ids:
        paths = (recordings / SPLIT / case_id).glob("trial_*.json")
        if trials := sorted(int(path.stem.removeprefix("trial_")) for path in paths):
            found[case_id] = trials
    return found


def replay(case: GoldCase, policy: Policy, recordings: Path, trial: int) -> TrialResult:
    """One recorded trial through the pipeline, scored against its gold case."""
    path = recording_path(recordings, SPLIT, case.case_id, trial)
    try:
        record = replay_trial(case, policy, path, DEFAULT_MODEL)
    except StaleRecording as error:
        return TrialResult(case_id=case.case_id, trial=trial, error=str(error))
    return TrialResult(case_id=case.case_id, trial=trial, checks=run_checks(case, policy, record), record=record)


def highlight(text: str, quotes: list[str]) -> str:
    """The note as safe HTML with each quoted passage marked.

    Text and quotes are compared after normalize(), as verify() compares them, so the note is shown in
    that form. Everything is escaped: a note is data and must never become markup.
    """
    shown = normalize(text)
    spans = []
    for quote in {normalize(q) for q in quotes if q.strip()}:
        start = shown.find(quote)
        while start != -1:
            spans.append((start, start + len(quote)))
            start = shown.find(quote, start + 1)
    merged: list[list[int]] = []
    for start, end in sorted(spans):
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    parts, position = [], 0
    for start, end in merged:
        parts += [html.escape(shown[position:start]), "<mark>", html.escape(shown[start:end]), "</mark>"]
        position = end
    return "".join([*parts, html.escape(shown[position:])])
