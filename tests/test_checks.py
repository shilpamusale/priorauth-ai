"""Each check must fail on the failure it exists to catch."""

from pathlib import Path

from evals.checks import (
    decision_follows_criteria,
    evidence_grounded,
    ready_requires_all_met,
    run_checks,
)
from evals.schema import load_gold_set
from priorauth.schemas import CriterionResult, CriterionStatus, Decision, Determination

GOLD = load_gold_set(Path(__file__).resolve().parents[1] / "evals" / "gold" / "cases.yaml")
READY = next(c for c in GOLD.cases if c.case_id == "lumbar_mri_ready_01")
INJECTION = next(c for c in GOLD.cases if c.case_id == "lumbar_mri_injection_01")
POLICY = GOLD.policy(READY.policy_id)


def met(cid, quote):
    return CriterionResult(criterion_id=cid, status=CriterionStatus.MET, evidence=[quote])


def not_met(cid):
    return CriterionResult(criterion_id=cid, status=CriterionStatus.NOT_MET)


def determination(case, decision, criteria):
    return Determination(
        case_id=case.case_id, policy_id=POLICY.policy_id, decision=decision, criteria=criteria, rationale="r"
    )


def test_correct_output_passes_every_check():
    out = determination(
        READY,
        Decision.READY_TO_SUBMIT,
        [
            met("symptom_duration", "low back pain radiating to the left leg for 10 weeks"),
            met("conservative_therapy", "Completed 8 weeks of physical therapy"),
        ],
    )
    assert all(r.passed for r in run_checks(READY, POLICY, out))


def test_invented_quote_fails_grounding():
    out = determination(
        READY,
        Decision.READY_TO_SUBMIT,
        [met("symptom_duration", "pain for 12 weeks"), met("conservative_therapy", "8 weeks of physical therapy")],
    )
    assert not evidence_grounded(READY, out).passed


def test_obeying_injected_note_fails_both_hard_gates():
    """The output a model produces if it follows the note in the clinical summary."""
    out = determination(
        INJECTION, Decision.READY_TO_SUBMIT, [not_met("symptom_duration"), not_met("conservative_therapy")]
    )
    assert not ready_requires_all_met(POLICY, out).passed
    assert not decision_follows_criteria(POLICY, out).passed
