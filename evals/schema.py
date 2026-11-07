"""Gold-set format. Expectations are written and committed before the agent runs on them."""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, Field, model_validator

from priorauth.decision import decide
from priorauth.schemas import CriterionResult, CriterionStatus, Decision, PARequest, Policy


class GoldCase(BaseModel):
    case_id: str
    tags: list[str] = Field(default_factory=list)
    policy_id: str
    request: PARequest
    expected_criteria: dict[str, CriterionStatus]
    expected_decision: Decision
    rationale: str  # why a reviewer would expect this outcome


class GoldSet(BaseModel):
    version: int
    policies: list[Policy]
    cases: list[GoldCase]

    def policy(self, policy_id: str) -> Policy:
        return next(p for p in self.policies if p.policy_id == policy_id)

    @model_validator(mode="after")
    def _self_consistent(self) -> GoldSet:
        """Every case must reference a known policy, label every criterion, and have an expected
        decision that follows from its expected criteria. Catches authoring errors before scoring."""
        ids = {p.policy_id for p in self.policies}
        for case in self.cases:
            if case.policy_id not in ids:
                raise ValueError(f"{case.case_id}: unknown policy {case.policy_id}")
            policy = self.policy(case.policy_id)
            if set(case.expected_criteria) != {c.criterion_id for c in policy.criteria}:
                raise ValueError(f"{case.case_id}: expected_criteria must label every policy criterion")
            results = [
                CriterionResult(criterion_id=cid, status=s, evidence=["<gold>"] if s is CriterionStatus.MET else [])
                for cid, s in case.expected_criteria.items()
            ]
            if decide(policy, results) is not case.expected_decision:
                raise ValueError(f"{case.case_id}: expected_decision contradicts expected_criteria")
        return self


def load_gold_set(path: Path) -> GoldSet:
    return GoldSet.model_validate(yaml.safe_load(path.read_text()))
