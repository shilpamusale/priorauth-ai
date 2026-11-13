"""Each contract must reject the malformed input it exists to stop."""

import datetime

import pytest
from pydantic import ValidationError

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

NOTE = {"note_id": "n1", "date": "2026-08-14", "note_type": "progress note", "text": "Axial low back pain, NRS 7/10."}
REQUEST = {"case_id": "c", "payer": "x", "service_code": "64493", "icd10_codes": ["M47.816"], "notes": [NOTE]}


@pytest.mark.parametrize("code", ["M54.16", "M47.816", "M1A.00X0", "Z3A.20", "C7A.00", "O9A.211", "S72.001A", "E11"])
def test_real_icd10_codes_are_accepted(code):
    assert PARequest(**{**REQUEST, "icd10_codes": [code]}).icd10_codes == [code]


@pytest.mark.parametrize("code", ["64493", "0213T", "3008F", "J3490", "E0601"])
def test_cpt_and_hcpcs_codes_are_accepted(code):
    assert PARequest(**{**REQUEST, "service_code": code}).service_code == code


@pytest.mark.parametrize(
    "field,value",
    [
        ("service_code", "6449"),
        ("service_code", "644930"),
        ("service_code", "Z9999"),
        ("icd10_codes", []),
        ("icd10_codes", ["54.50"]),
        ("icd10_codes", ["m54.16"]),
        ("icd10_codes", ["M54.12345"]),
        ("notes", []),
    ],
)
def test_malformed_requests_are_rejected(field, value):
    with pytest.raises(ValidationError):
        PARequest(**{**REQUEST, field: value})


def test_unknown_field_is_rejected():
    """A leftover field such as the old clinical_summary must fail, not silently vanish."""
    with pytest.raises(ValidationError, match="Extra inputs"):
        PARequest(**REQUEST, clinical_summary="s")


def test_note_date_is_parsed():
    assert PARequest(**REQUEST).notes[0].date == datetime.date(2026, 8, 14)


@pytest.mark.parametrize(
    "field,value",
    [("note_id", "Note 1"), ("note_id", ""), ("date", "2026-13-01"), ("note_type", " "), ("text", "   ")],
)
def test_malformed_notes_are_rejected(field, value):
    with pytest.raises(ValidationError):
        ClinicalNote(**{**NOTE, field: value})


def test_duplicate_note_ids_are_rejected():
    with pytest.raises(ValidationError, match="unique"):
        PARequest(**{**REQUEST, "notes": [NOTE, NOTE | {"text": "Another note."}]})


def test_criterion_is_required_by_default():
    assert PolicyCriterion(criterion_id="a", text="A").required is True


def test_policy_needs_at_least_one_criterion():
    with pytest.raises(ValidationError):
        Policy(policy_id="p", title="t", source="illustrative", criteria=[])


def test_met_without_evidence_is_rejected():
    with pytest.raises(ValidationError, match="must cite evidence"):
        CriterionResult(criterion_id="a", status=CriterionStatus.MET)


def test_met_with_evidence_names_its_note():
    result = CriterionResult(
        criterion_id="a", status=CriterionStatus.MET, evidence=[{"note_id": "n1", "quote": "NRS 7/10"}]
    )
    assert result.evidence == [Evidence(note_id="n1", quote="NRS 7/10")]


def test_evidence_without_a_note_is_rejected():
    with pytest.raises(ValidationError):
        CriterionResult(criterion_id="a", status=CriterionStatus.MET, evidence=[{"quote": "NRS 7/10"}])


@pytest.mark.parametrize("quote", ["", "   "])
def test_blank_quote_is_rejected(quote):
    with pytest.raises(ValidationError):
        Evidence(note_id="n1", quote=quote)


def test_not_met_needs_no_evidence():
    assert CriterionResult(criterion_id="a", status=CriterionStatus.NOT_MET).evidence == []


def test_needs_documentation_must_list_gaps():
    with pytest.raises(ValidationError, match="missing documentation"):
        Determination(case_id="c", policy_id="p", decision=Decision.NEEDS_DOCUMENTATION, criteria=[], rationale="r")


def test_blank_source_quote_is_rejected():
    with pytest.raises(ValidationError):
        PolicyCriterion(criterion_id="a", text="A", source_quotes=["  "])