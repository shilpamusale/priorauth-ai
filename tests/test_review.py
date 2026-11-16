"""Human review: the reviewer corrects facts with reasons and quotes; decide() still makes the decision."""

import datetime

import pytest
from pydantic import ValidationError

from priorauth.decision import decide
from priorauth.review import Override, Review, ReviewError, review_reasons, reviewed_determination, reviewed_findings
from priorauth.schemas import (
    ClinicalNote,
    CriterionResult,
    CriterionStatus,
    Decision,
    Determination,
    Evidence,
    PARequest,
    Policy,
    PolicyCriterion,
)

MET, NOT_MET, MISSING = CriterionStatus.MET, CriterionStatus.NOT_MET, CriterionStatus.INSUFFICIENT_EVIDENCE
POLICY = Policy(
    policy_id="p",
    title="t",
    source="illustrative",
    service_codes=["64493"],
    criteria=[
        PolicyCriterion(criterion_id="p.duration", text="A"),
        PolicyCriterion(criterion_id="p.imaging", text="B"),
    ],
)
REQUEST = PARequest(
    case_id="c",
    payer="Medicare",
    request_date="2026-09-01",
    service_code="64493",
    icd10_codes=["M47.816"],
    notes=[
        ClinicalNote(
            note_id="n1", date="2026-05-02", note_type="progress note", text="Low back pain since January 2026."
        ),
        ClinicalNote(
            note_id="n2", date="2026-08-20", note_type="MRI report", text="MRI of the lumbar spine: no fracture."
        ),
    ],
)
DURATION = CriterionResult(
    criterion_id="p.duration", status=MET, evidence=[Evidence(note_id="n1", quote="pain since January 2026")]
)
IMAGING_GAP = CriterionResult(criterion_id="p.imaging", status=MISSING, reasoning="No imaging found.")
FINDINGS = [IMAGING_GAP, DURATION]  # out of policy order on purpose


def review(**overrides):
    return Review(
        case_id="c",
        policy_id="p",
        trial=1,
        reviewer="Ishi",
        reviewed_at=datetime.datetime(2026, 11, 16, tzinfo=datetime.UTC),
        overrides=overrides,
    )


def test_with_no_overrides_the_findings_and_the_decision_stand():
    results = reviewed_findings(POLICY, REQUEST, FINDINGS, review())
    assert results == [DURATION, IMAGING_GAP]  # policy order
    assert reviewed_determination(POLICY, REQUEST, FINDINGS, review()).decision is decide(POLICY, FINDINGS)


def test_a_reviewer_quote_can_settle_a_gap_and_code_then_decides_ready():
    found = Override(status=MET, reason="The MRI is in n2.", evidence=[Evidence(note_id="n2", quote="no fracture")])
    determination = reviewed_determination(POLICY, REQUEST, FINDINGS, review(**{"p.imaging": found}))
    assert determination.decision is Decision.READY_TO_SUBMIT
    [imaging] = [r for r in determination.criteria if r.criterion_id == "p.imaging"]
    assert (imaging.status, imaging.reasoning) == (MET, "Reviewer: The MRI is in n2.")


def test_a_reviewer_can_overturn_a_met_finding_and_the_case_escalates():
    wrong = Override(status=NOT_MET, reason="January to September is under the policy's minimum here.")
    determination = reviewed_determination(POLICY, REQUEST, FINDINGS, review(**{"p.duration": wrong}))
    assert determination.decision is Decision.ESCALATE
    assert "Not met: p.duration." in determination.rationale


def test_reviewed_gaps_are_listed_as_missing_documentation():
    determination = reviewed_determination(POLICY, REQUEST, FINDINGS, review())
    assert determination.decision is Decision.NEEDS_DOCUMENTATION
    assert determination.missing_documentation == ["p.imaging: No imaging found."]


def test_marking_met_needs_a_quote():
    with pytest.raises(ValidationError, match="must quote the chart"):
        Override(status=MET, reason="I saw it.")


@pytest.mark.parametrize(
    "evidence,problem",
    [
        (Evidence(note_id="n2", quote="healed fracture"), "is not in n2"),
        (Evidence(note_id="n9", quote="no fracture"), "no note 'n9'"),
        (Evidence(note_id="n2", quote="fracture"), "too short"),
    ],
)
def test_a_reviewer_quote_is_checked_like_the_agents(evidence, problem):
    with pytest.raises(ReviewError, match=problem):
        reviewed_findings(
            POLICY, REQUEST, FINDINGS, review(**{"p.imaging": Override(status=MET, reason="r", evidence=[evidence])})
        )


def test_an_override_for_a_criterion_outside_the_policy_is_refused():
    with pytest.raises(ReviewError, match=r"not in p: \['q.other'\]"):
        reviewed_findings(POLICY, REQUEST, FINDINGS, review(**{"q.other": Override(status=NOT_MET, reason="r")}))


@pytest.mark.parametrize("field,value", [("case_id", "other"), ("policy_id", "other")])
def test_a_review_of_another_case_or_policy_is_refused(field, value):
    with pytest.raises(ReviewError, match="the review is of"):
        reviewed_findings(POLICY, REQUEST, FINDINGS, review().model_copy(update={field: value}))


def test_a_criterion_nobody_checked_stays_unchecked_and_escalates():
    assert reviewed_determination(POLICY, REQUEST, [DURATION], review()).decision is Decision.ESCALATE
    checked = Override(status=MISSING, reason="No imaging in the chart.")
    reviewed = reviewed_determination(POLICY, REQUEST, [DURATION], review(**{"p.imaging": checked}))
    assert reviewed.decision is Decision.NEEDS_DOCUMENTATION


def test_a_reviewer_cannot_set_the_decision():
    with pytest.raises(ValidationError, match="decision"):
        Review.model_validate(review().model_dump() | {"decision": "ready_to_submit"})


def determination(decision, **statuses):
    results = [
        CriterionResult(
            criterion_id=cid,
            status=s,
            evidence=[Evidence(note_id="n1", quote="pain since January")] if s is MET else [],
        )
        for cid, s in statuses.items()
    ]
    missing = [f"{cid}: gap" for cid, s in statuses.items() if s is MISSING]
    return Determination(
        case_id="c", policy_id="p", decision=decision, criteria=results, missing_documentation=missing, rationale="r"
    )


READY = determination(Decision.READY_TO_SUBMIT, **{"p.duration": MET, "p.imaging": MET})


def test_consistent_trials_that_do_not_escalate_need_no_review():
    assert review_reasons({1: READY, 2: READY, 3: READY}) == []


def test_an_escalation_needs_review():
    escalated = determination(Decision.ESCALATE, **{"p.duration": NOT_MET, "p.imaging": MET})
    assert review_reasons({1: escalated, 2: escalated}) == ["the agent escalated"]


def test_trials_that_disagree_need_review_and_say_where():
    gap = determination(Decision.NEEDS_DOCUMENTATION, **{"p.duration": MET, "p.imaging": MISSING})
    assert review_reasons({2: gap, 1: READY}) == [
        "trials reached different decisions: trial 1 ready_to_submit, trial 2 needs_documentation",
        "trials disagree on p.imaging: trial 1 met, trial 2 insufficient_evidence",
    ]


def test_a_finding_missing_from_one_trial_is_a_disagreement():
    partial = determination(Decision.ESCALATE, **{"p.duration": MET})
    assert "trials disagree on p.imaging: trial 1 met, trial 2 no finding" in review_reasons({1: READY, 2: partial})
