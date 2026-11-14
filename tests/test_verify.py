"""Only findings whose quotes are in the chart reach decide()."""

import datetime

from priorauth.decision import decide
from priorauth.schemas import (
    ClinicalNote,
    CriterionResult,
    CriterionStatus,
    Decision,
    Evidence,
    PARequest,
    Policy,
    PolicyCriterion,
)
from priorauth.verify import verify

POLICY = Policy(
    policy_id="demo",
    title="Demo policy",
    source="illustrative",
    criteria=[
        PolicyCriterion(criterion_id="demo.duration", text="Pain has lasted at least 3 months."),
        PolicyCriterion(criterion_id="demo.scale", text="A disability scale is recorded."),
    ],
)
REQUEST = PARequest(
    case_id="dev_900",
    payer="Medicare",
    request_date=datetime.date(2026, 10, 1),
    service_code="64493",
    icd10_codes=["M54.50"],
    notes=[
        ClinicalNote(
            note_id="n1",
            date=datetime.date(2026, 6, 1),
            note_type="primary care visit",
            text="Low back pain since March 2026. Started physical therapy.",
        ),
        ClinicalNote(
            note_id="n2",
            date=datetime.date(2026, 9, 1),
            note_type="physical therapy discharge summary",
            text="ODI 44% at discharge. No change with Physical   Therapy, per the “patient”.",
        ),
    ],
)
DURATION = CriterionResult(
    criterion_id="demo.duration",
    status=CriterionStatus.MET,
    evidence=[Evidence(note_id="n1", quote="Low back pain since March 2026")],
)
SCALE = CriterionResult(
    criterion_id="demo.scale",
    status=CriterionStatus.MET,
    evidence=[Evidence(note_id="n2", quote="ODI 44% at discharge")],
)


def scale_citing(*evidence):
    return SCALE.model_copy(update={"evidence": [Evidence(note_id=n, quote=q) for n, q in evidence]})


def test_grounded_findings_pass_through_in_order():
    result = verify(REQUEST, [DURATION, SCALE])
    assert result.grounded == [DURATION, SCALE]
    assert result.problems == []


def test_findings_without_evidence_have_nothing_to_check():
    gap = CriterionResult(criterion_id="demo.scale", status=CriterionStatus.INSUFFICIENT_EVIDENCE)
    assert verify(REQUEST, [gap]).grounded == [gap]


def test_quote_matching_ignores_spacing_and_curly_quotes():
    finding = scale_citing(("n2", 'No change with Physical Therapy, per the "patient"'))
    assert verify(REQUEST, [finding]).grounded == [finding]


def test_quote_matching_keeps_case():
    """A quote is verbatim: the matcher copies text, it does not paraphrase it."""
    result = verify(REQUEST, [scale_citing(("n2", "odi 44% at discharge"))])
    assert result.grounded == []


def test_each_kind_of_bad_quote_is_dropped_and_reported():
    findings = [
        scale_citing(("n2", "ODI 48% at discharge")),  # altered
        scale_citing(("n1", "ODI 44% at discharge")),  # real text, wrong note
        scale_citing(("n9", "ODI 44% at discharge")),  # note not in the chart
        scale_citing(("n2", "ODI")),  # too short to show anything
    ]
    result = verify(REQUEST, findings)
    assert result.grounded == []
    assert result.problems == [
        "demo.scale: quote 'ODI 48% at discharge' is not in n2",
        "demo.scale: quote 'ODI 44% at discharge' is not in n1",
        "demo.scale: no note 'n9' in this chart",
        "demo.scale: quote 'ODI' is too short; quote at least 2 words",
    ]


def test_one_bad_quote_drops_the_whole_finding():
    finding = scale_citing(("n2", "ODI 44% at discharge"), ("n2", "ODI 50% at intake"))
    result = verify(REQUEST, [DURATION, finding])
    assert result.grounded == [DURATION]
    assert len(result.problems) == 1


def test_an_invented_quote_sends_the_case_to_a_human():
    """Without the check this case would be ready to submit on evidence that is not in the chart."""
    invented = scale_citing(("n2", "ODI 52% at intake"))
    assert decide(POLICY, [DURATION, invented]) is Decision.READY_TO_SUBMIT
    assert decide(POLICY, verify(REQUEST, [DURATION, invented]).grounded) is Decision.ESCALATE