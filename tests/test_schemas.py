import pytest
from pydantic import ValidationError

from priorauth.schemas import CriterionResult, CriterionStatus, Decision, Determination, PARequest


def test_met_without_evidence_is_rejected():
    with pytest.raises(ValidationError, match="must cite evidence"):
        CriterionResult(criterion_id="a", status=CriterionStatus.MET)


def test_needs_documentation_must_list_gaps():
    with pytest.raises(ValidationError, match="missing documentation"):
        Determination(case_id="c", policy_id="p", decision=Decision.NEEDS_DOCUMENTATION, criteria=[], rationale="r")


@pytest.mark.parametrize("field,value", [("cpt_code", "7214"), ("icd10_codes", ["54.50"]), ("icd10_codes", [])])
def test_malformed_codes_are_rejected(field, value):
    request = {"case_id": "c", "payer": "x", "cpt_code": "72148", "icd10_codes": ["M54.50"], "clinical_summary": "s"}
    with pytest.raises(ValidationError):
        PARequest(**{**request, field: value})
