"""Human review: a reviewer confirms or corrects the agent's findings, and code decides again.

The reviewer changes facts, never the decision. Each override sets one criterion's status with a reason,
and decide() runs on the reviewed findings exactly as it runs on the agent's. A criterion the reviewer
marks met must quote the chart, and the quote is checked against its note like every agent quote.

A case needs a person when the agent escalated it, or when its trials disagree: two runs of the same
model on the same chart that reach different answers are a sign the chart or the policy is unclear.
"""

import datetime
from collections.abc import Mapping
from typing import Self

from pydantic import Field, model_validator

from priorauth.decision import decide
from priorauth.pipeline import missing_documentation, rationale
from priorauth.schemas import (
    Contract,
    CriterionResult,
    CriterionStatus,
    Decision,
    Determination,
    Evidence,
    NonBlank,
    PARequest,
    Policy,
)
from priorauth.verify import quote_problem


class ReviewError(ValueError):
    """A review that does not fit its case: another case or policy, an unknown criterion, a bad quote."""


class Override(Contract):
    """The reviewer's status for one criterion, and why."""

    status: CriterionStatus
    reason: NonBlank
    evidence: list[Evidence] = Field(default_factory=list)  # required when the reviewer marks it met

    @model_validator(mode="after")
    def _met_needs_evidence(self) -> Self:
        if self.status is CriterionStatus.MET and not self.evidence:
            raise ValueError("a criterion the reviewer marks met must quote the chart")
        return self


class Review(Contract):
    """One reviewer's corrections to one recorded run. There is no decision field: code decides."""

    case_id: str
    policy_id: str
    trial: int = Field(ge=1)  # the recorded run the reviewer looked at
    reviewer: NonBlank
    reviewed_at: datetime.datetime
    overrides: dict[str, Override] = Field(default_factory=dict)  # criteria left out are confirmed as found
    comment: str = ""


def reviewed_findings(
    policy: Policy, request: PARequest, findings: list[CriterionResult], review: Review
) -> list[CriterionResult]:
    """The agent's findings with the reviewer's overrides in their place, in policy order."""
    if (review.case_id, review.policy_id) != (request.case_id, policy.policy_id):
        raise ReviewError(
            f"the review is of {review.case_id} under {review.policy_id}, "
            f"not {request.case_id} under {policy.policy_id}"
        )
    if unknown := sorted(review.overrides.keys() - {c.criterion_id for c in policy.criteria}):
        raise ReviewError(f"overrides for criteria not in {policy.policy_id}: {unknown}")
    notes = {note.note_id: note for note in request.notes}
    for criterion_id, override in review.overrides.items():
        if problems := [problem for item in override.evidence if (problem := quote_problem(item, notes))]:
            raise ReviewError(f"{criterion_id}: {'; '.join(problems)}")

    by_id = {finding.criterion_id: finding for finding in findings}
    for criterion_id, override in review.overrides.items():
        by_id[criterion_id] = CriterionResult(
            criterion_id=criterion_id,
            status=override.status,
            evidence=override.evidence,
            reasoning=f"Reviewer: {override.reason}",
        )
    return [by_id[c.criterion_id] for c in policy.criteria if c.criterion_id in by_id]


def reviewed_determination(
    policy: Policy, request: PARequest, findings: list[CriterionResult], review: Review
) -> Determination:
    """decide() on the reviewed findings: the reviewer settles the facts, code settles the decision."""
    results = reviewed_findings(policy, request, findings, review)
    found = {r.criterion_id for r in results}
    skipped = [c.criterion_id for c in policy.criteria if c.criterion_id not in found]
    decision = decide(policy, results)
    return Determination(
        case_id=request.case_id,
        policy_id=policy.policy_id,
        decision=decision,
        criteria=results,
        missing_documentation=missing_documentation(policy, results),
        rationale=rationale(decision, policy, results, skipped, []),
    )


def review_reasons(trials: Mapping[int, Determination]) -> list[str]:
    """Why a case needs a person: the agent escalated, or its trials disagree. Empty when neither."""
    ordered = sorted(trials.items())
    reasons = []
    decisions = [determination.decision for _, determination in ordered]
    if Decision.ESCALATE in decisions:
        reasons.append("the agent escalated")
    if len(set(decisions)) > 1:
        reasons.append(
            "trials reached different decisions: " + ", ".join(f"trial {t} {d.decision}" for t, d in ordered)
        )
    statuses: dict[str, dict[int, CriterionStatus]] = {}
    for trial, determination in ordered:
        for result in determination.criteria:
            statuses.setdefault(result.criterion_id, {})[trial] = result.status
    for criterion_id, by_trial in statuses.items():
        seen = [by_trial.get(trial) for trial, _ in ordered]
        if len(set(seen)) > 1:
            reasons.append(
                f"trials disagree on {criterion_id}: "
                + ", ".join(f"trial {t} {by_trial.get(t, 'no finding')}" for t, _ in ordered)
            )
    return reasons
