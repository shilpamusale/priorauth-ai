"""Typed contracts for every stage of the prior-authorization pipeline.

Every handoff is validated against these models, so a malformed result fails
loudly instead of flowing downstream.
"""

import datetime
from enum import StrEnum
from typing import Annotated, Self

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    model_validator,
)

ICD10Code = Annotated[str, StringConstraints(pattern=r"^[A-Z]\d[0-9A-Z](\.[0-9A-Z]{1,4})?$")]
ServiceCode = Annotated[str, StringConstraints(pattern=r"^(\d{4}[0-9A-Z]|[A-V]\d{4})$")]  # CPT or HCPCS Level II
NoteId = Annotated[str, StringConstraints(pattern=r"^[a-z0-9_-]+$")]
NonBlank = Annotated[str, StringConstraints(pattern=r"\S")]
Quote = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]


class Contract(BaseModel):
    """Base for every contract: an unknown field is an error,
    not something to ignore."""

    model_config = ConfigDict(extra="forbid")


class CriterionStatus(StrEnum):
    MET = "met"
    NOT_MET = "not_met"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"


class Decision(StrEnum):
    READY_TO_SUBMIT = "ready_to_submit"  # every required criterion is met and cited.
    NEEDS_DOCUMENTATION = "needs_documentation"  # nothing contradicts the policy, but evidence is missing.
    ESCALATE = "escalate"  # a required criterion is not met or was not checked: a human decision needed.


class ClinicalNote(Contract):
    """One document in the chart: a visit note, an imaging report, a therapy summary."""

    note_id: NoteId
    date: datetime.date
    note_type: NonBlank
    text: NonBlank


class PARequest(Contract):
    """What the provider wants to submit: the ordered service
    and the patient's chart, as separate notes."""

    case_id: str
    payer: str
    request_date: datetime.date  # durations such as "pain for 3 months" are measured up to this date
    service_code: ServiceCode
    icd10_codes: list[ICD10Code] = Field(min_length=1)
    notes: list[ClinicalNote] = Field(min_length=1)

    @model_validator(mode="after")
    def _unique_note_ids(self) -> Self:
        ids = [note.note_id for note in self.notes]
        if len(ids) != len(set(ids)):
            raise ValueError("note IDs must be unique within a chart")
        return self
    
    @model_validator(mode="after")
    def _no_note_after_the_request(self) -> Self:
        if late := [note.note_id for note in self.notes if note.date > self.request_date]:
            raise ValueError(f"notes dated after the request: {late}")
        return self


class PolicyCriterion(Contract):
    criterion_id: str
    text: str
    required: bool = True  # safe default: a criterion only stops counting if a policy says so
    source_quotes: list[Quote] = Field(default_factory=list)  # verbatim from the coverage document; see policies.py


class Policy(Contract):
    policy_id: str
    title: str
    source: str  # URL of the coverage document, or "illustrative" for test fixtures
    criteria: list[PolicyCriterion] = Field(min_length=1)


class Evidence(Contract):
    """A verbatim quote from the chart and the note it comes from."""

    note_id: NoteId
    quote: Quote


class CriterionResult(Contract):
    """The matcher's judgement on one criterion."""

    criterion_id: str
    status: CriterionStatus
    evidence: list[Evidence] = Field(default_factory=list)  # the pipeline checks each quote against its note
    reasoning: str = ""

    @model_validator(mode="after")
    def _met_needs_evidence(self) -> Self:
        if self.status is CriterionStatus.MET and not self.evidence:
            raise ValueError(f"{self.criterion_id}: a criterion marked met must cite evidence.")
        return self


class Determination(Contract):
    """Final pipeline output, written to the audit trail."""

    case_id: str
    policy_id: str
    decision: Decision
    criteria: list[CriterionResult]
    missing_documentation: list[str] = Field(default_factory=list)
    rationale: str

    @model_validator(mode="after")
    def _gap_listed(self) -> Self:
        if self.decision is Decision.NEEDS_DOCUMENTATION and not self.missing_documentation:
            raise ValueError("needs_documentation must list the missing documentation")
        return self