"""The documentation agent's runner: record and replay its requests on dev cases with gaps."""

import datetime
import json

import pytest
import yaml

from evals.gaps import GapResult, documentation_path, failures, main, report, run_cost, run_gaps
from evals.gold import GoldCase, Label
from evals.runner import Mode, record_trial, recording_path
from priorauth.llm.base import ModelResponse, StopReason, ToolCall, Usage
from priorauth.llm.scripted import ScriptedClient
from priorauth.schemas import ClinicalNote, CriterionStatus, Decision, PARequest, Policy, PolicyCriterion

POLICY = Policy(
    policy_id="demo",
    title="Demo policy",
    source="illustrative",
    service_codes=["64493"],
    criteria=[
        PolicyCriterion(criterion_id="demo.duration", text="Pain has lasted at least 3 months."),
        PolicyCriterion(criterion_id="demo.scale", text="A disability scale is recorded."),
    ],
)
POLICIES = {"demo": POLICY}
NOTE_TEXT = "Low back pain since March 2026. No disability scale recorded yet."


def case(case_id, scale_status, decision):
    request = PARequest(
        case_id=case_id,
        payer="Medicare",
        request_date=datetime.date(2026, 10, 1),
        service_code="64493",
        ordered_service="Lumbar medial branch blocks",
        icd10_codes=["M54.50"],
        notes=[ClinicalNote(note_id="n1", date=datetime.date(2026, 6, 1), note_type="visit", text=NOTE_TEXT)],
    )
    scale = Label(status=scale_status, evidence_notes=["n1"] if scale_status is CriterionStatus.MET else [])
    labels = {"demo.duration": Label(status=CriterionStatus.MET, evidence_notes=["n1"]), "demo.scale": scale}
    return GoldCase(policy_id="demo", request=request, labels=labels, expected_decision=decision)


@pytest.fixture
def gold(tmp_path):
    """dev_900 has a documentation gap; dev_901 has none."""
    cases = [
        case("dev_900", CriterionStatus.INSUFFICIENT_EVIDENCE, Decision.NEEDS_DOCUMENTATION),
        case("dev_901", CriterionStatus.MET, Decision.READY_TO_SUBMIT),
    ]
    for c in cases:
        path = tmp_path / "gold" / "dev" / f"{c.case_id}.yaml"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(yaml.safe_dump(c.model_dump(mode="json")))
    return tmp_path / "gold", {c.case_id: c for c in cases}


def evidence_reply(scale_status, reasoning="No disability scale recorded."):
    duration = {
        "criterion_id": "demo.duration",
        "status": "met",
        "evidence": [{"note_id": "n1", "quote": "Low back pain since March 2026"}],
        "reasoning": "Since March.",
    }
    scale = {"criterion_id": "demo.scale", "status": scale_status, "evidence": [], "reasoning": reasoning}
    if scale_status == "met":
        scale["evidence"] = [{"note_id": "n1", "quote": "Low back pain since March 2026"}]
    calls = [ToolCall(id=f"toolu_{i}", name="submit_finding", input=f) for i, f in enumerate([duration, scale])]
    return ModelResponse(tool_calls=calls, stop_reason=StopReason.TOOL_USE)


def record_evidence(gold_cases, recordings, reasoning="No disability scale recorded."):
    """The single agent's first trial for each case, as the documentation agent's input."""
    for case_id, status in [("dev_900", "insufficient_evidence"), ("dev_901", "met")]:
        client = ScriptedClient([evidence_reply(status, reasoning)])
        record_trial(
            gold_cases[case_id], POLICY, client, recording_path(recordings, "dev", case_id, 1), "claude-sonnet-5-5"
        )


REQUEST = {
    "criterion_id": "demo.scale",
    "request": "Please record a baseline disability scale, such as the Oswestry Disability Index, at the next visit.",
    "note_type": "next visit note",
}


def documentation_model():
    call = ToolCall(id="toolu_1", name="write_request", input=REQUEST)
    usage = Usage(input_tokens=3000, output_tokens=500)
    return ScriptedClient([ModelResponse(tool_calls=[call], stop_reason=StopReason.TOOL_USE, usage=usage)])


def test_record_then_replay_the_requests(gold, tmp_path):
    root, gold_cases = gold
    record_evidence(gold_cases, tmp_path)
    recorded = run_gaps(Mode.RECORD, POLICIES, client=documentation_model(), gold=root, recordings=tmp_path)
    assert [r.case_id for r in recorded] == ["dev_900"]  # dev_901 has no gaps, so no model call
    assert documentation_path(tmp_path, "dev_900").exists()
    assert not documentation_path(tmp_path, "dev_901").exists()
    replayed = run_gaps(Mode.REPLAY, POLICIES, gold=root, recordings=tmp_path)
    assert replayed[0].run.requests == recorded[0].run.requests
    assert replayed[0].run.requests[0].note_type == "next visit note"
    assert failures(replayed) == []


def test_new_evidence_makes_the_documentation_recording_stale(gold, tmp_path):
    root, gold_cases = gold
    record_evidence(gold_cases, tmp_path)
    run_gaps(Mode.RECORD, POLICIES, client=documentation_model(), gold=root, recordings=tmp_path)
    record_evidence(gold_cases, tmp_path, reasoning="The chart has no ODI or other disability scale.")
    [result] = run_gaps(Mode.REPLAY, POLICIES, gold=root, recordings=tmp_path)
    assert result.error == "model call 1 differs from the recording"
    assert failures([result]) == ["dev_900: model call 1 differs from the recording"]


def test_missing_recordings_are_errors_not_crashes(gold, tmp_path):
    root, gold_cases = gold
    results = run_gaps(Mode.REPLAY, POLICIES, gold=root, recordings=tmp_path)  # no evidence step recorded yet
    assert [r.case_id for r in results] == ["dev_900", "dev_901"]  # without it, gaps are unknown for every case
    assert all(r.error.startswith("no recording at") for r in results)
    record_evidence(gold_cases, tmp_path)
    [result] = run_gaps(Mode.REPLAY, POLICIES, gold=root, recordings=tmp_path)
    assert result.error.endswith("documentation/dev/dev_900/trial_1.json")


def test_a_gap_left_without_a_request_fails_the_gate(gold, tmp_path):
    root, gold_cases = gold
    record_evidence(gold_cases, tmp_path)
    silent = ScriptedClient([ModelResponse(stop_reason=StopReason.OTHER)])
    [result] = run_gaps(Mode.RECORD, POLICIES, client=silent, gold=root, recordings=tmp_path)
    assert result.run.skipped == ["demo.scale"]
    assert failures([result]) == ["dev_900: no request for demo.scale"]
    assert "gaps left without a request: demo.scale" in report(result)


def test_report_and_cost(gold, tmp_path):
    root, gold_cases = gold
    record_evidence(gold_cases, tmp_path)
    [result] = run_gaps(Mode.RECORD, POLICIES, client=documentation_model(), gold=root, recordings=tmp_path)
    assert report(result).splitlines() == ["dev_900: 1 request", f"  - scale (next visit note): {REQUEST['request']}"]
    assert run_cost(result.run) == pytest.approx(0.011)  # 3,000 in at $2/M + 500 out at $10/M
    assert report(GapResult(case_id="dev_900", error="no recording")) == "dev_900: ERROR no recording"


def test_bad_arguments_are_refused(gold, tmp_path):
    root, _ = gold
    with pytest.raises(ValueError, match="needs a live model client"):
        run_gaps(Mode.RECORD, POLICIES, gold=root, recordings=tmp_path)
    with pytest.raises(ValueError, match="no such cases"):
        run_gaps(Mode.REPLAY, POLICIES, gold=root, recordings=tmp_path, cases=["dev_999"])


def test_command_line_replay(gold, tmp_path, monkeypatch, capsys):
    root, gold_cases = gold
    monkeypatch.setattr("evals.gaps.load_policies", lambda directory: POLICIES)
    args = ["replay", "--gold", str(root), "--recordings", str(tmp_path)]
    record_evidence(gold_cases, tmp_path)
    assert main(args) == 1  # no documentation recording yet
    run_gaps(Mode.RECORD, POLICIES, client=documentation_model(), gold=root, recordings=tmp_path)
    assert main(args) == 0
    out = capsys.readouterr().out
    assert "1 cases with gaps, 1 requests, 0 errors, cost $0.01" in out and out.rstrip().endswith("gate passed")


def test_a_documentation_recording_with_unused_replies_is_stale(gold, tmp_path):
    root, gold_cases = gold
    record_evidence(gold_cases, tmp_path)
    run_gaps(Mode.RECORD, POLICIES, client=documentation_model(), gold=root, recordings=tmp_path)
    path = documentation_path(tmp_path, "dev_900")
    data = json.loads(path.read_text())
    data["exchanges"].append(data["exchanges"][-1])
    path.write_text(json.dumps(data))
    [result] = run_gaps(Mode.REPLAY, POLICIES, gold=root, recordings=tmp_path)
    assert result.error == "the run made 1 of the 2 recorded model calls"
