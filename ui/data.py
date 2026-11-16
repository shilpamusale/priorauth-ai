"""Data for the demo: recorded trials, replayed through the real pipeline.

The demo never calls a model. Each page replays a recording of a live run through the same match, verify
and decide code that CI tests, so what it shows is what the code does with those model replies. Only dev
cases appear; held-out cases stay unseen until the final evaluation. The LLM judge's grades are replayed
from its recordings in the same way.
"""

import html
import os
from collections.abc import Mapping
from pathlib import Path

from pydantic import ValidationError

from evals.calibration import CalibrationReport
from evals.checks import run_checks
from evals.gold import GoldCase
from evals.judge import Grade, run_judge
from evals.judge_items import ItemKind, build_items
from evals.routing import RoutingResult, routing_cases
from evals.routing import run as run_routing
from evals.runner import GOLD, POLICIES, RECORDINGS, Mode, TrialResult, pipeline_root, recording_path, replay_trial
from priorauth.ingest import normalize
from priorauth.llm.base import ToolResult
from priorauth.llm.replay import StaleRecording
from priorauth.matcher import DEFAULT_MODEL
from priorauth.pipeline import Pipeline
from priorauth.policies import load_sources
from priorauth.retrieval import PolicyIndex, lcd_name
from priorauth.review import Override
from priorauth.schemas import CriterionResult, Evidence, Policy, PolicyCriterion

SPLIT = "dev"
KEEP = "keep the agent's finding"  # the review form's first choice for each criterion
CALIBRATION = Path("evals/results/judge_calibration.json")


def folders() -> tuple[Path, Path, Path]:
    """Gold cases, recordings and policies. Environment variables override the defaults (used by tests)."""
    return (
        Path(os.environ.get("PRIORAUTH_GOLD", GOLD)),
        Path(os.environ.get("PRIORAUTH_RECORDINGS", RECORDINGS)),
        Path(os.environ.get("PRIORAUTH_POLICIES", POLICIES)),
    )


def overrides_from_answers(
    case_id: str, criteria: list[PolicyCriterion], findings: dict[str, CriterionResult], answers: Mapping[str, str]
) -> tuple[dict[str, Override], list[str]]:
    """The reviewer's changes from the review form's answers (keyed case:criterion:field), and what is wrong."""
    overrides, problems = {}, []
    for criterion in criteria:
        cid = criterion.criterion_id
        choice = answers[f"{case_id}:{cid}:status"]
        reason = answers[f"{case_id}:{cid}:reason"].strip()
        note, quote = answers[f"{case_id}:{cid}:note"], answers[f"{case_id}:{cid}:quote"].strip()
        if choice == KEEP:
            if cid not in findings:
                problems.append(f"{cid}: the agent left this unchecked, so give it a status")
            continue
        if not reason:
            problems.append(f"{cid}: say why you changed it")
            continue
        evidence = [Evidence(note_id=note, quote=quote)] if note and quote else []
        try:
            overrides[cid] = Override(status=choice, reason=reason, evidence=evidence)
        except ValidationError:
            problems.append(f"{cid}: marking it met needs a quote from a note")
    return overrides, problems


def build_index(policy_dir: Path) -> PolicyIndex:
    """The retrieval index over every LCD text in policy_dir/sources. Tests replace it, so they never need the
    embedding model; the demo image has the model built in."""
    return PolicyIndex(load_sources(policy_dir))


def dev_routes(policies: dict[str, Policy], gold: Path, index: PolicyIndex) -> list[RoutingResult]:
    """Retrieval and the code check on every dev request: the gold cases and the routing-only cases."""
    return run_routing(routing_cases(gold, SPLIT, policies), index, policies)


def route_rows(results: list[RoutingResult]) -> list[dict]:
    """One table row per routed request: what it should reach, what it reached, and how close the LCD was."""
    rows = []
    for r in results:
        best = r.route.candidates[0] if r.route.candidates else None
        rows.append(
            {
                "case": r.case.case_id,
                "order": r.case.ordered_service,
                "code": r.case.service_code,
                "should reach": r.case.policy_id or r.case.expected.value,
                "reached": r.route.policy_id or r.route.outcome.value,
                "right": "yes" if r.correct else "UNSAFE" if r.unsafe else "no, escalated",
                "closest LCD": lcd_name(best.source) if best else "",
                "distance": round(best.distance, 3) if best else None,
            }
        )
    return rows


def recorded_trials(recordings: Path, case_ids: list[str]) -> dict[str, list[int]]:
    """Trial numbers that have a recording, for each case with at least one."""
    found = {}
    for case_id in case_ids:
        paths = (recordings / SPLIT / case_id).glob("trial_*.json")
        if trials := sorted(int(path.stem.removeprefix("trial_")) for path in paths):
            found[case_id] = trials
    return found


def replay(
    case: GoldCase, policy: Policy, recordings: Path, trial: int, pipeline: Pipeline = Pipeline.SINGLE_AGENT
) -> TrialResult:
    """One recorded trial through the pipeline that made it, scored against its gold case."""
    path = recording_path(pipeline_root(recordings, pipeline), SPLIT, case.case_id, trial)
    try:
        record = replay_trial(case, policy, path, DEFAULT_MODEL, pipeline)
    except StaleRecording as error:
        return TrialResult(case_id=case.case_id, trial=trial, error=str(error))
    return TrialResult(case_id=case.case_id, trial=trial, checks=run_checks(case, policy, record), record=record)


def replay_all(
    policies: dict[str, Policy],
    cases: dict[str, GoldCase],
    recordings: Path,
    pipeline: Pipeline = Pipeline.SINGLE_AGENT,
) -> dict[tuple[str, int], TrialResult]:
    """Every recorded trial of one pipeline, replayed and scored, keyed by (case ID, trial number) in case order."""
    return {
        (case_id, trial): replay(cases[case_id], policies[cases[case_id].policy_id], recordings, trial, pipeline)
        for case_id, trials in recorded_trials(pipeline_root(recordings, pipeline), list(cases)).items()
        for trial in trials
    }


def judge_grades(policies: dict[str, Policy], gold: Path, recordings: Path) -> dict[tuple[ItemKind, str, str], Grade]:
    """The judge's recorded grades, keyed by (kind, case ID, criterion ID): evidence items and requests only.

    The judge graded the single agent's first trial and the documentation agent's requests. If those runs
    no longer replay, there is nothing to attach grades to and the page shows none; a missing or stale judge
    recording leaves out that one grade.
    """
    try:
        items, cases = build_items(policies, gold, recordings)
    except ValueError:
        return {}
    graded = {i.item_id: i for i in items if i.kind is not ItemKind.PLANTED}
    return {
        (graded[j.item_id].kind, graded[j.item_id].case_id, graded[j.item_id].criterion_id): j.grade
        for j in run_judge(Mode.REPLAY, list(graded.values()), cases, recordings=recordings)
        if j.grade
    }


def calibration(path: Path = CALIBRATION) -> CalibrationReport | None:
    """The committed report of the judge's agreement with the hand labels, if there is one."""
    try:
        return CalibrationReport.model_validate_json(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def judge_note(report: CalibrationReport | None, model: str) -> str:
    """What the judge is, how far it agreed with a person, and where it is known to be wrong."""
    note = f"An LLM judge ({model}) graded the evidence for each met criterion in this trial."
    if report and report.evidence.kappa is not None:
        agreed = report.evidence
        note += (
            f" Against blind hand labels it agreed on {agreed.agreed} of {agreed.items} items "
            f"(Cohen's kappa {agreed.kappa:.2f})."
        )
    return note + (
        " It is more lenient than a person on partial evidence, and it is not given the request date, so its "
        "grades for pain duration can be wrong."
    )


def brief(result: ToolResult) -> str:
    """A tool result shortened for the run log: errors in full, anything else to its first line and sentence.

    The note text a read_note call returned is already shown in the chart, and a submit_finding reply repeats
    the list of criteria still open, so the log keeps only what identifies the step.
    """
    if result.is_error:
        return f"Error: {result.content}"
    return result.content.splitlines()[0].split(". ")[0] if result.content else ""


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
