"""Chart tools: reading notes, searching the chart, and recording findings with checked quotes."""

import datetime

import pytest

from priorauth.llm.base import ToolCall
from priorauth.schemas import (
    ClinicalNote,
    CriterionResult,
    CriterionStatus,
    Evidence,
    PARequest,
    Policy,
    PolicyCriterion,
)
from priorauth.tools import MAX_SEARCH_HITS, ChartTools

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


def tools() -> ChartTools:
    return ChartTools(POLICY, REQUEST)


def call(name, **args):
    return ToolCall(id="toolu_1", name=name, input=args)


def submit(criterion_id="demo.duration", status="met", evidence=None, reasoning="Documented."):
    if evidence is None:
        evidence = [{"note_id": "n1", "quote": "Low back pain since March 2026"}]
    return call("submit_finding", criterion_id=criterion_id, status=status, evidence=evidence, reasoning=reasoning)


def test_note_index_lists_notes_without_their_text():
    index = tools().note_index()
    assert index.splitlines() == [
        "n1 | 2026-06-01 | primary care visit",
        "n2 | 2026-09-01 | physical therapy discharge summary",
    ]
    assert "ODI" not in index


def test_read_note_returns_the_note():
    result = tools().run(call("read_note", note_id="n2"))
    assert not result.is_error
    assert result.tool_call_id == "toolu_1"
    assert result.content.startswith("n2 | 2026-09-01 | physical therapy discharge summary\n\n")
    assert "ODI 44% at discharge." in result.content


def test_read_note_rejects_an_unknown_note():
    result = tools().run(call("read_note", note_id="n9"))
    assert result.is_error
    assert "no note 'n9'" in result.content and "n1, n2" in result.content


def test_search_ignores_case_and_spacing_and_names_each_note():
    result = tools().run(call("search_chart", query="physical  therapy"))
    assert not result.is_error
    lines = result.content.splitlines()
    assert [line.split(":")[0] for line in lines] == ["n1", "n2"]
    assert "Physical Therapy" in lines[1]  # the note's own text, so it can be quoted


def test_search_shows_only_text_near_the_match():
    note = REQUEST.notes[0].model_copy(update={"text": "a" * 300 + " ODI 52% " + "b" * 300})
    request = REQUEST.model_copy(update={"notes": [note]})
    line = ChartTools(POLICY, request).run(call("search_chart", query="odi")).content
    assert line.startswith("n1: ...") and line.endswith("...")
    assert "ODI 52%" in line and len(line) < 200


def test_search_with_no_match_is_not_an_error():
    result = tools().run(call("search_chart", query="radiculopathy"))
    assert not result.is_error
    assert result.content == "no matches for 'radiculopathy'"


def test_search_caps_the_number_of_matches():
    note = REQUEST.notes[0].model_copy(update={"text": "pain. " * (MAX_SEARCH_HITS + 5)})
    request = REQUEST.model_copy(update={"notes": [note]})
    lines = ChartTools(POLICY, request).run(call("search_chart", query="pain")).content.splitlines()
    assert len(lines) == MAX_SEARCH_HITS + 1
    assert lines[-1] == f"(first {MAX_SEARCH_HITS} of {MAX_SEARCH_HITS + 5} matches)"


def test_search_rejects_a_blank_query():
    assert tools().run(call("search_chart", query="  ")).is_error


def test_submit_records_the_finding_and_names_what_is_left():
    chart = tools()
    result = chart.run(submit())
    assert not result.is_error
    assert result.content == "Recorded demo.duration. Still to submit: demo.scale."
    assert chart.findings["demo.duration"].evidence == [Evidence(note_id="n1", quote="Low back pain since March 2026")]
    assert chart.remaining() == ["demo.scale"]


def test_submit_says_when_every_criterion_is_in():
    chart = tools()
    chart.run(submit())
    result = chart.run(submit("demo.scale", status="insufficient_evidence", evidence=[]))
    assert result.content == "Recorded demo.scale. All criteria are submitted."
    assert chart.remaining() == []
    assert chart.findings["demo.scale"].status is CriterionStatus.INSUFFICIENT_EVIDENCE


def test_quote_matching_ignores_spacing_and_curly_quotes():
    evidence = [{"note_id": "n2", "quote": 'No change with Physical Therapy, per the "patient"'}]
    assert not tools().run(submit("demo.scale", evidence=evidence)).is_error


def test_criterion_outside_the_policy_is_rejected():
    chart = tools()
    result = chart.run(submit("other_policy.duration"))
    assert result.is_error and "not a criterion of this policy" in result.content
    assert chart.findings == {}


def test_a_criterion_is_submitted_only_once():
    chart = tools()
    chart.run(submit())
    result = chart.run(submit(status="not_met", evidence=[]))
    assert result.is_error and "already has a finding" in result.content
    assert chart.findings["demo.duration"].status is CriterionStatus.MET


@pytest.mark.parametrize(
    "evidence,problem",
    [
        ([{"note_id": "n1", "quote": "ODI 44% at discharge"}], "is not in n1"),  # real quote, wrong note
        ([{"note_id": "n2", "quote": "ODI 48% at discharge"}], "is not in n2"),  # altered quote
        ([{"note_id": "n1", "quote": "pain"}], "too short"),
        ([{"note_id": "n9", "quote": "Low back pain"}], "no note 'n9'"),
    ],
)
def test_bad_evidence_is_rejected_and_nothing_is_recorded(evidence, problem):
    chart = tools()
    result = chart.run(submit(evidence=evidence))
    assert result.is_error and problem in result.content
    assert chart.findings == {}


def test_every_bad_quote_is_reported_at_once():
    evidence = [{"note_id": "n1", "quote": "pain"}, {"note_id": "n2", "quote": "ODI 48% at discharge"}]
    result = tools().run(submit(evidence=evidence))
    assert "too short" in result.content and "is not in n2" in result.content


def test_met_without_evidence_is_rejected():
    result = tools().run(submit(evidence=[]))
    assert result.is_error and "must cite evidence" in result.content


@pytest.mark.parametrize(
    "bad_call",
    [
        ToolCall(id="toolu_1", name="delete_chart", input={}),
        ToolCall(id="toolu_1", name="read_note", input={}),
        ToolCall(id="toolu_1", name="read_note", input={"note_id": 5}),
        ToolCall(id="toolu_1", name="search_chart", input={"query": "pain", "limit": 3}),
        ToolCall(id="toolu_1", name="submit_finding", input={"criterion_id": "demo.duration", "status": "probably"}),
    ],
)
def test_malformed_calls_come_back_as_errors(bad_call):
    result = tools().run(bad_call)
    assert result.is_error and result.tool_call_id == "toolu_1"


def strict_objects(schema):
    """Every object schema inside a tool's input schema."""
    if isinstance(schema, dict):
        if schema.get("type") == "object":
            yield schema
        for value in schema.values():
            yield from strict_objects(value)
    elif isinstance(schema, list):
        for value in schema:
            yield from strict_objects(value)


def test_schemas_have_the_form_strict_tool_use_expects():
    for spec in tools().specs():
        assert spec.strict
        for obj in strict_objects(spec.input_schema):
            assert obj["additionalProperties"] is False, spec.name
            assert obj["required"] == list(obj["properties"]), spec.name


def test_submit_schema_binds_input_to_this_policy_and_chart():
    schema = {spec.name: spec for spec in tools().specs()}["submit_finding"].input_schema
    props = schema["properties"]
    assert props["criterion_id"]["enum"] == ["demo.duration", "demo.scale"]
    assert props["status"]["enum"] == ["met", "not_met", "insufficient_evidence"]
    assert props["evidence"]["items"]["properties"]["note_id"]["enum"] == ["n1", "n2"]


def test_submit_schema_matches_the_result_contract():
    """If CriterionResult or Evidence gains or loses a field, the tool schema must change with it."""
    schema = {spec.name: spec for spec in tools().specs()}["submit_finding"].input_schema
    assert set(schema["properties"]) == set(CriterionResult.model_fields)
    assert set(schema["properties"]["evidence"]["items"]["properties"]) == set(Evidence.model_fields)