"""Gold cases: synthetic charts with hand labels, written before the agent runs on them.

Each case is one YAML file in evals/gold/<split>, named after its case ID. 
The dev split is used while building; the held-out split stays untouched until
the final run. Every rule here runs when the cases load, so a labelling mistake 
fails before any scoring.
"""

from pathlib import Path
from typing import Self

import yaml
from pydantic import Field, ValidationError, model_validator

from priorauth.decision import decide
from priorauth.schemas import Contract, CriterionResult, CriterionStatus, Decision, Evidence, NoteId, PARequest, Policy

SPLITS = ("dev", "heldout")

class GoldError(ValueError):
    """A gold case is malformed or inconsistent with its policy."""

class Label(Contract):
    """The hand label for one criterion."""

    status: CriterionStatus
    evidence_notes: list[NoteId] = Field(default_factory=list) # the notes that decide this label

    @model_validator(mode="after")
    def _met_names_its_notes(self) -> Self:
        if self.status is CriterionStatus.MET and not self.evidence_notes:
            raise ValueError("a met label must name the notes that hold its evidence")
        return self

class GoldCase(Contract):
    """One synthetic request with a label for every criterion of its policy."""

    policy_id: str
    tags: list[str] = Field(default_factory=list)
    request: PARequest
    labels: dict[str, Label]
    expected_decision: Decision

    @property
    def case_id(self) -> str:
        return self.request.case_id

    def as_results(self) -> list[CriterionResult]:
        """The labels as criterion results, so decide() can check the expected decision."""
        return [
            CriterionResult(
                criterion_id=cid,
                status=label.status,
                evidence=[Evidence(note_id=note_id, quote="(gold label)") for note_id in label.evidence_notes],
            )
            for cid, label in self.labels.items()
        ]

def load_case(path: Path, policies: dict[str, Policy]) -> GoldCase:
    """Load one case and check it against its policy."""
    try:
        case = GoldCase.model_validate(yaml.safe_load(path.read_text()))
    except (yaml.YAMLError, ValidationError) as e:
        raise GoldError(f"{path.name}: {e}") from e

    if case.case_id != path.stem:
        raise GoldError(f"{path.name}: case_id {case.case_id!r} must match the file name")

    if (policy:= policies.get(case.policy_id)) is None:
        raise GoldError(f"{path.name}: unknown policy {case.policy_id!r}")

    criteria = {c.criterion_id for c in policy.criteria}
    if missing := sorted(criteria - case.labels.keys()):
        raise GoldError(f"{path.name}: criteria without a label: {missing}")
    if extra := sorted(case.labels.keys() - criteria):
        raise GoldError(f"{path.name}: labels for criteria not in {policy.policy_id} : {extra}")

    notes = {note.note_id for note in case.request.notes}
    if unknown := sorted({n for label in case.labels.values() for n in label.evidence_notes} - notes):
        raise GoldError(f"{path.name}: evidence_notes not in the chart: {unknown}")

    if (implied := decide(policy, case.as_results())) is not case.expected_decision:
        raise GoldError(f"{path.name}: expected_decision is {case.expected_decision}, but the labels give {implied}")
    return case

def load_split(root: Path, split: str, policies: dict[str, Policy]) -> dict[str, GoldCase]:
    """Load every case in one split, keyed by case ID (unique because case IDs equal file names)."""
    if split not in SPLITS:
        raise GoldError(f"unknown split {split!r}; use one of {SPLITS}")
    paths = sorted((root / split).glob("*.yaml"))
    if not paths:
        raise GoldError(f"no gold cases in {root / split}")
    return {case.case_id: case for case in (load_case(path, policies) for path in paths)}  