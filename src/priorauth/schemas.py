"""Typed contracts for every stage of the prior-authorization pipeline.

The model's structured output is validated against these schemas at each handoff, so a malformed
or ungrounded result fails loudly instead of flowing downstream.
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field, model_validator


class CriterionStatus(str, Enum):
    MET = "met"
    NOT_MET = "not_met"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"


class Decision(str, Enum):
    READY_TO_SUBMIT = "ready_to_submit"  # every required criterion is met and cited
    NEEDS_DOCUMENTATION = "needs_documentation"  # nothing contradicts the policy, but evidence is missing
    ESCALATE = "escalate"  # a required criterion is not met, or the case is ambiguous: a human decides


class PARequest(BaseModel):
    """A provider-side request: the ordered service and the patient's clinical summary."""

    case_id: str
    payer: str
    cpt_code: str = Field(pattern=r"^\d{4}[0-9A-Z]$")
    icd10_codes: list[str] = Field(min_length=1)
    clinical_summary: str

    @model_validator(mode="after")
    def _icd10_format(self) -> PARequest:
        for code in self.icd10_codes:
            if not (len(code) >= 3 and code[0].isalpha() and code[1:3].isdigit()):
                raise ValueError(f"not an ICD-10-CM code: {code!r}")
        return self


class PolicyCriterion(BaseModel):
    criterion_id: str
    text: str
    required: bool = True


class Policy(BaseModel):
    policy_id: str
    title: str
    source: str  # URL of the coverage document, or "illustrative" for test fixtures
    criteria: list[PolicyCriterion] = Field(min_length=1)


class CriterionResult(BaseModel):
    criterion_id: str
    status: CriterionStatus
    evidence: list[str] = Field(default_factory=list)  # verbatim quotes from the clinical summary
    note: str = ""

    @model_validator(mode="after")
    def _met_needs_evidence(self) -> CriterionResult:
        if self.status is CriterionStatus.MET and not self.evidence:
            raise ValueError(f"{self.criterion_id}: a criterion marked met must cite evidence")
        return self


class Determination(BaseModel):
    """Final pipeline output, written to the audit trail."""

    case_id: str
    policy_id: str
    decision: Decision
    criteria: list[CriterionResult]
    missing_documentation: list[str] = Field(default_factory=list)
    rationale: str

    @model_validator(mode="after")
    def _gaps_listed(self) -> Determination:
        if self.decision is Decision.NEEDS_DOCUMENTATION and not self.missing_documentation:
            raise ValueError("needs_documentation must list the missing documentation")
        return self
