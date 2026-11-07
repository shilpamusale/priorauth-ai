"""Deterministic checks applied to every pipeline output. Hard gates fail CI on any single trial."""

from __future__ import annotations

import re
from dataclasses import dataclass

from evals.schema import GoldCase
from priorauth.decision import decide
from priorauth.schemas import CriterionStatus, Decision, Determination, Policy

HARD_GATES = {"ready_requires_all_met", "decision_follows_criteria"}


@dataclass
class CheckResult:
    name: str
    passed: bool
    detail: str = ""


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().lower()


def decision_correct(case: GoldCase, out: Determination) -> CheckResult:
    ok = out.decision is case.expected_decision
    return CheckResult("decision_correct", ok, "" if ok else f"{out.decision.value} != {case.expected_decision.value}")


def criteria_correct(case: GoldCase, out: Determination) -> CheckResult:
    got = {r.criterion_id: r.status for r in out.criteria}
    wrong = [
        f"{cid}: {got.get(cid, 'missing')} != {want.value}"
        for cid, want in case.expected_criteria.items()
        if got.get(cid) is not want
    ]
    return CheckResult("criteria_correct", not wrong, "; ".join(str(w) for w in wrong))


def evidence_grounded(case: GoldCase, out: Determination) -> CheckResult:
    """Every quote must appear verbatim (up to whitespace and case) in the clinical summary."""
    summary = _norm(case.request.clinical_summary)
    invented = [q for r in out.criteria for q in r.evidence if _norm(q) not in summary]
    return CheckResult("evidence_grounded", not invented, "; ".join(f"not in summary: {q!r}" for q in invented))


def ready_requires_all_met(policy: Policy, out: Determination) -> CheckResult:
    """Safety: a case is never marked ready to submit while a required criterion is unmet."""
    if out.decision is not Decision.READY_TO_SUBMIT:
        return CheckResult("ready_requires_all_met", True)
    got = {r.criterion_id: r.status for r in out.criteria}
    unmet = [
        c.criterion_id for c in policy.criteria if c.required and got.get(c.criterion_id) is not CriterionStatus.MET
    ]
    return CheckResult("ready_requires_all_met", not unmet, f"ready_to_submit with unmet: {unmet}" if unmet else "")


def decision_follows_criteria(policy: Policy, out: Determination) -> CheckResult:
    """The output's decision must be the one its own criterion results imply (no bypassing decide())."""
    implied = decide(policy, out.criteria)
    ok = out.decision is implied
    return CheckResult(
        "decision_follows_criteria", ok, "" if ok else f"{out.decision.value}, criteria imply {implied.value}"
    )


def run_checks(case: GoldCase, policy: Policy, out: Determination) -> list[CheckResult]:
    return [
        decision_correct(case, out),
        criteria_correct(case, out),
        evidence_grounded(case, out),
        ready_requires_all_met(policy, out),
        decision_follows_criteria(policy, out),
    ]
