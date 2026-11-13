"""Each contract must reject the malformed input it exists to stop."""

import pytest
from pydantic import ValidationError

from priorauth.schemas import (
    CriterionResult,
    CriterionStatus,
    Decision,
    Determination,
    PARequest,
    Policy,
    PolicyCriterion,
)

REQUEST = {"case_id": "c", "payer": "x", "cpt_code": "64493", "icd10_codes": ["M47.816"], "clinical_summary": "s"}


@pytest.mark.parametrize("code", ["M54.16", "M47.816", "M1A.00X0", "Z3A.20", "C7A.00", "O9A.211", "S72.001A", "E11"])
def test_real_icd10_codes_are_accepted(code):
    assert PARequest(**{**REQUEST, "icd10_codes": [code]}).icd10_codes == [code]

@pytest.mark.parametrize(
    "field,value",
    [
        ("cpt_code", "6449"),
        ("icd10_codes", []),
        ("icd10_codes",["54.50"]),
        ("icd10_codes", ["m54.16"]),
        ("icd10_codes", ["M54.12345"]),
    ],
)
def test_malformed_codes_are_rejected(field, value):
    with pytest.raises(ValidationError):
        PARequest(**{**REQUEST, field:value})

def test_unknown_field_is_rejected():
    with pytest.raises(ValidationError, match="Extra inputs"):
        PARequest(**REQUEST, surprise="x")

def test_criterion_is_required_by_default():
    assert PolicyCriterion(criterion_id="a", text="A").required is True

def test_policy_needs_at_least_one_criterion():
    with pytest.raises(ValidationError):
        Policy(policy_id="p", title="t", source="illustrative", criteria=[])

def test_met_without_evidence_is_rejected():
    with pytest.raises(ValidationError, match="must cite evidence"):
        CriterionResult(criterion_id="a", status=CriterionStatus.MET)

@pytest.mark.parametrize("quote", ["", "  "])
def test_blank_quote_is_rejected(quote):
    with pytest.raises(ValidationError):
        CriterionResult(criterion_id="a", status=CriterionStatus.MET, evidence=[quote])

def test_not_met_need_no_evidence():
    assert CriterionResult(criterion_id="a", status=CriterionStatus.NOT_MET).evidence == []

def test_needs_documentation_must_list_gaps():
    with pytest.raises(ValidationError, match="missing documentation"):
        Determination(case_id="c", policy_id="p", decision=Decision.NEEDS_DOCUMENTATION, criteria=[], rationale="r")

def test_blank_source_quote_is_rejected():
    with pytest.raises(ValidationError):
        PolicyCriterion(criterion_id="a", text="A", source_quotes=["  "])
