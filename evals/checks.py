"""Checks that score one audit record against its gold case.

Quality checks feed the metrics. Hard gates are safety properties: one failure on any trial fails the CI
gate, whatever the averages say. The gates do not trust the pipeline: each one recomputes its property
from the recorded output, so a bug in the pipeline, or in decide() itself, cannot hide a failure.
"""

from evals.gold import GoldCase
from priorauth.decision import decide
from priorauth.pipeline import AuditRecord
from priorauth.schemas import Contract, CriterionStatus, Decision, Policy
from priorauth.verify import quote_problem


class CheckResult(Contract):
    name: str
    passed: bool
    hard_gate: bool = False
    detail: str = ""  # what failed, for the report


def statuses(record: AuditRecord) -> dict[str, CriterionStatus]:
    return {r.criterion_id: r.status for r in record.determination.criteria}


def decision_correct(case: GoldCase, record: AuditRecord) -> CheckResult:
    got, expected = record.determination.decision, case.expected_decision
    detail = "" if got is expected else f"got {got}, expected {expected}"
    return CheckResult(name="decision_correct", passed=got is expected, detail=detail)


def criteria_correct(case: GoldCase, record: AuditRecord) -> CheckResult:
    """Every criterion has the labelled status. A criterion with no finding counts as wrong."""
    got = statuses(record)
    wrong = [
        f"{cid}: got {got.get(cid, 'no finding')}, expected {label.status}"
        for cid, label in case.labels.items()
        if got.get(cid) is not label.status
    ]
    return CheckResult(name="criteria_correct", passed=not wrong, detail="; ".join(wrong))


def evidence_notes_match(case: GoldCase, record: AuditRecord) -> CheckResult:
    """Where the label and the run both say met, the run cites at least one of the notes the label names."""
    off = []
    for result in record.determination.criteria:
        label = case.labels.get(result.criterion_id)
        if label and label.status is CriterionStatus.MET and result.status is CriterionStatus.MET:
            cited = sorted({e.note_id for e in result.evidence})
            if not set(cited) & set(label.evidence_notes):
                off.append(f"{result.criterion_id}: cites {cited}, label names {label.evidence_notes}")
    return CheckResult(name="evidence_notes_match", passed=not off, detail="; ".join(off))


def evidence_grounded(case: GoldCase, record: AuditRecord) -> CheckResult:
    """Hard gate: every quote in the determination is in the note it names."""
    notes = {note.note_id: note for note in case.request.notes}
    bad = [
        f"{result.criterion_id}: {problem}"
        for result in record.determination.criteria
        for item in result.evidence
        if (problem := quote_problem(item, notes))
    ]
    return CheckResult(name="evidence_grounded", passed=not bad, hard_gate=True, detail="; ".join(bad))


def ready_requires_all_met(policy: Policy, record: AuditRecord) -> CheckResult:
    """Hard gate: a case is never ready to submit while a required criterion lacks a met finding."""
    if record.determination.decision is not Decision.READY_TO_SUBMIT:
        return CheckResult(name="ready_requires_all_met", passed=True, hard_gate=True)
    got = statuses(record)
    required = [c.criterion_id for c in policy.criteria if c.required]
    unmet = [cid for cid in required if got.get(cid) is not CriterionStatus.MET]
    detail = f"ready_to_submit with unmet: {', '.join(unmet)}" if unmet else ""
    return CheckResult(name="ready_requires_all_met", passed=not unmet, hard_gate=True, detail=detail)


def decision_follows_criteria(policy: Policy, record: AuditRecord) -> CheckResult:
    """Hard gate: the recorded decision is the one decide() gives for the recorded criteria."""
    got, implied = record.determination.decision, decide(policy, record.determination.criteria)
    detail = "" if got is implied else f"recorded {got}, criteria imply {implied}"
    return CheckResult(name="decision_follows_criteria", passed=got is implied, hard_gate=True, detail=detail)


def run_checks(case: GoldCase, policy: Policy, record: AuditRecord) -> list[CheckResult]:
    """Every check for one trial of one case."""
    ids = (record.determination.case_id, record.determination.policy_id, policy.policy_id)
    if ids != (case.case_id, case.policy_id, case.policy_id):
        raise ValueError(f"record {ids[:2]} and policy {ids[2]!r} do not belong to case {case.case_id!r}")
    return [
        decision_correct(case, record),
        criteria_correct(case, record),
        evidence_notes_match(case, record),
        evidence_grounded(case, record),
        ready_requires_all_met(policy, record),
        decision_follows_criteria(policy, record),
    ]