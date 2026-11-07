"""The decision rule: the safest outcome wins whenever evidence is incomplete."""

from priorauth.decision import decide
from priorauth.schemas import CriterionResult, CriterionStatus, Decision, Policy, PolicyCriterion

MET, NOT_MET, GAP = CriterionStatus.MET, CriterionStatus.NOT_MET, CriterionStatus.INSUFFICIENT_EVIDENCE

POLICY = Policy(
    policy_id="p",
    title="t",
    source="illustrative",
    criteria=[
        PolicyCriterion(criterion_id="a", text="A"),
        PolicyCriterion(criterion_id="b", text="B"),
        PolicyCriterion(criterion_id="optional", text="C", required=False),
    ],
)


def result(cid, status):
    return CriterionResult(criterion_id=cid, status=status, evidence=["quote"] if status is MET else [])


def test_all_required_met_is_ready():
    assert decide(POLICY, [result("a", MET), result("b", MET)]) is Decision.READY_TO_SUBMIT


def test_optional_criterion_does_not_block():
    assert decide(POLICY, [result("a", MET), result("b", MET), result("optional", NOT_MET)]) is Decision.READY_TO_SUBMIT


def test_missing_evidence_needs_documentation():
    assert decide(POLICY, [result("a", MET), result("b", GAP)]) is Decision.NEEDS_DOCUMENTATION


def test_not_met_escalates_even_with_gaps():
    assert decide(POLICY, [result("a", NOT_MET), result("b", GAP)]) is Decision.ESCALATE


def test_skipped_required_criterion_escalates():
    assert decide(POLICY, [result("a", MET)]) is Decision.ESCALATE
