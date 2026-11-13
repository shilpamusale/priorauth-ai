"""The decision rule: whenever the results are incomplete or inconsistent, the safest outcome wins."""

from priorauth.decision import decide
from priorauth.schemas import CriterionResult, CriterionStatus, Decision, Evidence, Policy, PolicyCriterion

MET, NOT_MET, GAP = CriterionStatus.MET, CriterionStatus.NOT_MET, CriterionStatus.INSUFFICIENT_EVIDENCE
EVIDENCE = [Evidence(note_id="n1", quote="quote")]

POLICY = Policy(
    policy_id = "p",
    title="t",
    source="illustrative",
    criteria=[
        PolicyCriterion(criterion_id="a", text="A"),
        PolicyCriterion(criterion_id="b", text="B"),
        PolicyCriterion(criterion_id="optional", text="C", required=False),
    ],
)

def result(cid, status):
    return CriterionResult(criterion_id=cid, status=status, evidence=EVIDENCE if status is MET else [])

def test_all_required_met_is_ready():
    assert decide(POLICY, [result("a", MET), result("b", MET)]) is Decision.READY_TO_SUBMIT


def test_result_order_does_not_matter():
    assert decide(POLICY, [result("b", MET), result("a", MET)]) is Decision.READY_TO_SUBMIT


def test_optional_criterion_does_not_block():
    assert decide(POLICY, [result("a", MET), result("b", MET), result("optional", NOT_MET)]) is Decision.READY_TO_SUBMIT


def test_missing_evidence_needs_documentation():
    assert decide(POLICY, [result("a", MET), result("b", GAP)]) is Decision.NEEDS_DOCUMENTATION


def test_not_met_escalates_even_with_gaps():
    assert decide(POLICY, [result("a", NOT_MET), result("b", GAP)]) is Decision.ESCALATE


def test_skipped_required_criterion_escalates():
    assert decide(POLICY, [result("a", MET)]) is Decision.ESCALATE


def test_no_results_escalates():
    assert decide(POLICY, []) is Decision.ESCALATE


def test_conflicting_duplicate_escalates():
    """Without this rule, the later 'met' would overwrite the earlier 'not_met'."""
    results = [result("a", NOT_MET), result("a", MET), result("b", MET)]
    assert decide(POLICY, results) is Decision.ESCALATE


def test_repeated_duplicate_escalates():
    results = [result("a", MET), result("a", MET), result("b", MET)]
    assert decide(POLICY, results) is Decision.ESCALATE


def test_result_for_unknown_criterion_escalates():
    results = [result("a", MET), result("b", MET), result("from_another_policy", MET)]
    assert decide(POLICY, results) is Decision.ESCALATE


def test_policy_without_required_criteria_escalates():
    policy = Policy(
        policy_id="p",
        title="t",
        source="illustrative",
        criteria=[PolicyCriterion(criterion_id="optional", text="C", required=False)],
    )
    assert decide(policy, [result("optional", MET)]) is Decision.ESCALATE