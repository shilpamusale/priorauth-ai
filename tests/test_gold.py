"""Gold cases are checked when they load, so a labelling mistake fails before any scoring."""

import pytest
import yaml

from evals.gold import GoldError, load_case, load_split
from priorauth.schemas import Decision, Policy, PolicyCriterion

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
POLICIES = {"p": POLICY}


def case_data(case_id="case_ready_01", **changes):
    data = {
        "policy_id": "p",
        "tags": ["clean"],
        "request": {
            "case_id": case_id,
            "payer": "Medicare",
            "request_date": "2026-09-01",
            "service_code": "64493",
            "ordered_service": "Lumbar medial branch blocks, L4-L5 and L5-S1",
            "icd10_codes": ["M47.816"],
            "notes": [
                {"note_id": "n1", "date": "2026-05-02", "note_type": "progress note", "text": "Pain since January."},
                {"note_id": "n2", "date": "2026-08-20", "note_type": "MRI report", "text": "No fracture."},
            ],
        },
        "labels": {
            "p.duration": {"status": "met", "evidence_notes": ["n1"]},
            "p.imaging": {"status": "met", "evidence_notes": ["n2"]},
        },
        "expected_decision": "ready_to_submit",
    }
    return data | changes


def write(directory, data, name=None):
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{name or data['request']['case_id']}.yaml"
    path.write_text(yaml.safe_dump(data))
    return path


def test_valid_case_loads(tmp_path):
    case = load_case(write(tmp_path, case_data()), POLICIES)
    assert case.case_id == "case_ready_01"
    assert case.expected_decision is Decision.READY_TO_SUBMIT
    assert case.labels["p.imaging"].evidence_notes == ["n2"]


def test_case_id_must_match_the_file_name(tmp_path):
    with pytest.raises(GoldError, match="must match the file name"):
        load_case(write(tmp_path, case_data(), name="renamed"), POLICIES)


def test_unknown_policy_is_rejected(tmp_path):
    with pytest.raises(GoldError, match="unknown policy"):
        load_case(write(tmp_path, case_data(policy_id="other")), POLICIES)


def test_every_criterion_needs_a_label(tmp_path):
    labels = {"p.duration": {"status": "met", "evidence_notes": ["n1"]}}
    with pytest.raises(GoldError, match=r"criteria without a label: \['p.imaging'\]"):
        load_case(write(tmp_path, case_data(labels=labels)), POLICIES)


def test_label_for_a_criterion_outside_the_policy_is_rejected(tmp_path):
    labels = case_data()["labels"] | {"q.duration": {"status": "not_met"}}
    with pytest.raises(GoldError, match="not in p"):
        load_case(write(tmp_path, case_data(labels=labels)), POLICIES)


def test_met_label_must_name_its_notes(tmp_path):
    labels = case_data()["labels"] | {"p.imaging": {"status": "met"}}
    with pytest.raises(GoldError, match="must name the notes"):
        load_case(write(tmp_path, case_data(labels=labels)), POLICIES)


def test_evidence_note_must_be_in_the_chart(tmp_path):
    labels = case_data()["labels"] | {"p.imaging": {"status": "met", "evidence_notes": ["n9"]}}
    with pytest.raises(GoldError, match=r"not in the chart: \['n9'\]"):
        load_case(write(tmp_path, case_data(labels=labels)), POLICIES)


def test_not_met_label_may_name_the_note_that_rules_it_out(tmp_path):
    labels = case_data()["labels"] | {"p.imaging": {"status": "not_met", "evidence_notes": ["n2"]}}
    case = load_case(write(tmp_path, case_data(labels=labels, expected_decision="escalate")), POLICIES)
    assert case.expected_decision is Decision.ESCALATE


def test_expected_decision_must_follow_from_the_labels(tmp_path):
    labels = case_data()["labels"] | {"p.imaging": {"status": "insufficient_evidence"}}
    with pytest.raises(GoldError, match="the labels give needs_documentation"):
        load_case(write(tmp_path, case_data(labels=labels)), POLICIES)


def test_malformed_case_names_the_file(tmp_path):
    data = case_data()
    del data["expected_decision"]
    with pytest.raises(GoldError, match="case_ready_01.yaml"):
        load_case(write(tmp_path, data), POLICIES)


def test_split_loads_keyed_by_case_id(tmp_path):
    write(tmp_path / "dev", case_data("case_ready_01"))
    write(tmp_path / "dev", case_data("case_ready_02"))
    write(tmp_path / "heldout", case_data("case_ready_03"))
    assert sorted(load_split(tmp_path, "dev", POLICIES)) == ["case_ready_01", "case_ready_02"]
    assert sorted(load_split(tmp_path, "heldout", POLICIES)) == ["case_ready_03"]


def test_unknown_split_is_rejected(tmp_path):
    with pytest.raises(GoldError, match="unknown split"):
        load_split(tmp_path, "test", POLICIES)


def test_empty_split_is_an_error(tmp_path):
    with pytest.raises(GoldError, match="no gold cases"):
        load_split(tmp_path, "dev", POLICIES)


def test_every_gold_case_needs_an_ordered_service(tmp_path):
    data = case_data()
    del data["request"]["ordered_service"]
    with pytest.raises(GoldError, match="needs an ordered_service"):
        load_case(write(tmp_path, data), POLICIES)


def test_the_service_code_must_be_one_the_policy_covers(tmp_path):
    data = case_data()
    data["request"]["service_code"] = "15823"
    with pytest.raises(GoldError, match="15823 is not one p covers"):
        load_case(write(tmp_path, data), POLICIES)
