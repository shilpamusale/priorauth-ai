"""The runner: record trials with a live client, replay them without one, and report what is missing."""

import datetime
import json
from pathlib import Path

import pytest
import yaml

import priorauth.llm.providers as providers
from evals.gold import GoldCase, Label
from evals.runner import Mode, TrialResult, main, recording_path, run_split, summary_line
from evals.scoring import Summary
from priorauth.llm.base import ModelResponse, StopReason, ToolCall, Usage
from priorauth.llm.providers import Provider
from priorauth.llm.scripted import ScriptedClient
from priorauth.pipeline import Pipeline
from priorauth.policies import load_policies
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


def case(case_id):
    request = PARequest(
        case_id=case_id,
        payer="Medicare",
        request_date=datetime.date(2026, 10, 1),
        service_code="64493",
        ordered_service="Lumbar medial branch blocks",
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
                text="ODI 44% at discharge. No change with physical therapy.",
            ),
        ],
    )
    met = CriterionStatus.MET
    labels = {
        "demo.duration": Label(status=met, evidence_notes=["n1"]),
        "demo.scale": Label(status=met, evidence_notes=["n2"]),
    }
    return GoldCase(policy_id="demo", request=request, labels=labels, expected_decision=Decision.READY_TO_SUBMIT)


@pytest.fixture
def gold(tmp_path):
    """A gold folder with two dev cases and one held-out case."""
    for split, case_id in [("dev", "dev_900"), ("dev", "dev_901"), ("heldout", "heldout_900")]:
        path = tmp_path / "gold" / split / f"{case_id}.yaml"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(yaml.safe_dump(case(case_id).model_dump(mode="json")))
    return tmp_path / "gold"


def live_replies(runs):
    """What a live model says on each run: read a note, then submit both findings."""
    read = ToolCall(id="toolu_1", name="read_note", input={"note_id": "n2"})
    duration = {
        "criterion_id": "demo.duration",
        "status": "met",
        "evidence": [{"note_id": "n1", "quote": "Low back pain since March 2026"}],
        "reasoning": "Pain since March.",
    }
    scale = {
        "criterion_id": "demo.scale",
        "status": "met",
        "evidence": [{"note_id": "n2", "quote": "ODI 44% at discharge"}],
        "reasoning": "ODI recorded.",
    }
    submits = [
        ToolCall(id="toolu_2", name="submit_finding", input=duration),
        ToolCall(id="toolu_3", name="submit_finding", input=scale),
    ]
    usage = Usage(input_tokens=900, output_tokens=120)
    one_run = [
        ModelResponse(tool_calls=[read], stop_reason=StopReason.TOOL_USE, usage=usage),
        ModelResponse(tool_calls=submits, stop_reason=StopReason.TOOL_USE, usage=usage),
    ]
    return one_run * runs


def record(gold, recordings, trials=2, cases=None):
    runs = trials * (len(cases) if cases else 2)
    return run_split(
        "dev",
        Mode.RECORD,
        POLICIES,
        client=ScriptedClient(live_replies(runs)),
        gold=gold,
        recordings=recordings,
        trials=trials,
        cases=cases,
    )


def test_record_saves_one_recording_per_trial(gold, tmp_path):
    results = record(gold, tmp_path / "rec")
    assert [(r.case_id, r.trial) for r in results] == [("dev_900", 1), ("dev_900", 2), ("dev_901", 1), ("dev_901", 2)]
    assert all(r.record and all(c.passed for c in r.checks) for r in results)
    for r in results:
        assert recording_path(tmp_path / "rec", "dev", r.case_id, r.trial).exists()


def test_replay_reproduces_the_recorded_trials_without_a_model(gold, tmp_path):
    recorded = record(gold, tmp_path / "rec")
    replayed = run_split("dev", Mode.REPLAY, POLICIES, gold=gold, recordings=tmp_path / "rec", trials=2)
    assert [r.record.determination for r in replayed] == [r.record.determination for r in recorded]
    assert [r.checks for r in replayed] == [r.checks for r in recorded]


def test_missing_recordings_are_reported_per_trial(gold, tmp_path):
    record(gold, tmp_path / "rec", trials=1, cases=["dev_900"])
    results = run_split("dev", Mode.REPLAY, POLICIES, gold=gold, recordings=tmp_path / "rec", trials=2)
    assert [bool(r.error) for r in results] == [False, True, True, True]
    assert results[1].error.startswith("no recording at") and results[1].error.endswith("dev_900/trial_2.json")


def test_a_changed_case_makes_its_recording_stale(gold, tmp_path):
    record(gold, tmp_path / "rec", trials=1)
    path = gold / "dev" / "dev_901.yaml"
    path.write_text(path.read_text().replace("ODI 44% at discharge.", "ODI 46% at discharge."))
    results = run_split("dev", Mode.REPLAY, POLICIES, gold=gold, recordings=tmp_path / "rec", trials=1)
    assert results[0].error == ""
    assert results[1].error == "model call 2 differs from the recording"


def test_replaying_as_another_model_is_stale(gold, tmp_path):
    record(gold, tmp_path / "rec", trials=1, cases=["dev_900"])
    result = run_split(
        "dev",
        Mode.REPLAY,
        POLICIES,
        gold=gold,
        recordings=tmp_path / "rec",
        trials=1,
        cases=["dev_900"],
        model="claude-haiku-5-5",
    )
    assert result[0].error == "model call 1 differs from the recording"


def test_a_recording_with_unused_replies_is_stale(gold, tmp_path):
    record(gold, tmp_path / "rec", trials=1, cases=["dev_900"])
    path = recording_path(tmp_path / "rec", "dev", "dev_900", 1)
    data = json.loads(path.read_text())
    data["exchanges"].append(data["exchanges"][-1])
    path.write_text(json.dumps(data))
    result = run_split(
        "dev", Mode.REPLAY, POLICIES, gold=gold, recordings=tmp_path / "rec", trials=1, cases=["dev_900"]
    )
    assert result[0].error == "the run made 2 of the 3 recorded model calls"


def test_held_out_cases_need_an_explicit_flag(gold, tmp_path):
    with pytest.raises(ValueError, match="final evaluation"):
        run_split("heldout", Mode.REPLAY, POLICIES, gold=gold, recordings=tmp_path)
    results = run_split("heldout", Mode.REPLAY, POLICIES, gold=gold, recordings=tmp_path, trials=1, allow_heldout=True)
    assert [r.case_id for r in results] == ["heldout_900"]


def test_each_trial_is_reported_as_soon_as_it_ends(gold, tmp_path):
    seen = []
    results = run_split(
        "dev",
        Mode.RECORD,
        POLICIES,
        client=ScriptedClient(live_replies(2)),
        gold=gold,
        recordings=tmp_path / "rec",
        trials=1,
        on_result=lambda result: seen.append((result.case_id, len(list((tmp_path / "rec").rglob("*.json"))))),
    )
    assert seen == [("dev_900", 1), ("dev_901", 2)]  # each reported right after its own recording was saved
    assert len(results) == 2


def test_bad_arguments_are_refused(gold, tmp_path):
    with pytest.raises(ValueError, match="no such cases in dev"):
        run_split("dev", Mode.REPLAY, POLICIES, gold=gold, recordings=tmp_path, cases=["dev_999"])
    with pytest.raises(ValueError, match="needs a live model client"):
        run_split("dev", Mode.RECORD, POLICIES, gold=gold, recordings=tmp_path)


def test_a_trial_has_a_record_or_an_error_but_not_both():
    with pytest.raises(ValueError):
        TrialResult(case_id="dev_900", trial=1)


def test_summary_lines(gold, tmp_path):
    ok = record(gold, tmp_path / "rec", trials=1, cases=["dev_900"])[0]
    assert summary_line(ok) == "dev_900 trial 1: ready_to_submit, all checks passed"
    failing = ok.model_copy(
        update={"checks": [c.model_copy(update={"passed": c.name != "evidence_grounded"}) for c in ok.checks]}
    )
    assert summary_line(failing) == "dev_900 trial 1: ready_to_submit, failed evidence_grounded (hard gate)"
    assert (
        summary_line(TrialResult(case_id="dev_900", trial=2, error="no recording"))
        == "dev_900 trial 2: ERROR no recording"
    )


def test_command_line_prints_the_summary_and_applies_the_gate(gold, tmp_path, capsys, monkeypatch):
    monkeypatch.setattr("evals.runner.load_policies", lambda directory: POLICIES)
    record(gold, tmp_path / "rec", trials=1)
    args = ["replay", "--gold", str(gold), "--recordings", str(tmp_path / "rec")]
    summary_file = tmp_path / "summary.json"
    assert main([*args, "--trials", "1", "--min-decision-accuracy", "1.0", "--summary", str(summary_file)]) == 0
    out = capsys.readouterr().out
    assert "dev_900 trial 1: ready_to_submit, all checks passed" in out
    assert "decision accuracy 1.00" in out and out.rstrip().endswith("gate passed")
    assert Summary.model_validate_json(summary_file.read_text()).trials == 2
    assert main([*args, "--trials", "2", "--cases", "dev_901"]) == 1
    out = capsys.readouterr().out
    assert "dev_901 trial 2: ERROR no recording" in out
    assert out.rstrip().endswith("gate failed: 1 trials have no record")


def test_command_line_applies_the_accuracy_threshold(gold, tmp_path, capsys, monkeypatch):
    monkeypatch.setattr("evals.runner.load_policies", lambda directory: POLICIES)
    record(gold, tmp_path / "rec", trials=1)
    # Relabel one case. Labels are not sent to the model, so its recording still replays.
    path = gold / "dev" / "dev_901.yaml"
    data = yaml.safe_load(path.read_text())
    data["labels"]["demo.scale"] = {"status": "insufficient_evidence", "evidence_notes": []}
    data["expected_decision"] = "needs_documentation"
    path.write_text(yaml.safe_dump(data))
    args = ["replay", "--gold", str(gold), "--recordings", str(tmp_path / "rec"), "--trials", "1"]
    assert main([*args, "--min-decision-accuracy", "0.5"]) == 0
    assert main([*args, "--min-decision-accuracy", "0.6"]) == 1
    assert capsys.readouterr().out.rstrip().endswith("gate failed: decision accuracy 0.50 is below 0.60")


def test_the_real_dev_split_loads(tmp_path):
    """With no recordings yet, every dev case is reported, none crashes."""
    results = run_split("dev", Mode.REPLAY, load_policies(Path("policies")), recordings=tmp_path, trials=1)
    assert len(results) == len(list(Path("evals/gold/dev").glob("*.yaml")))
    assert all(r.error.startswith("no recording at") for r in results)


class EvidenceModel:
    """Answers each evidence agent for the one criterion its tools allow, so parallel call order does not matter."""

    QUOTES = {"demo.duration": ("n1", "Low back pain since March 2026"), "demo.scale": ("n2", "ODI 44% at discharge")}

    def complete(self, request):
        tool = next(t for t in request.tools if t.name == "submit_finding")
        [criterion_id] = tool.input_schema["properties"]["criterion_id"]["enum"]
        note_id, quote = self.QUOTES[criterion_id]
        finding = {
            "criterion_id": criterion_id,
            "status": "met",
            "evidence": [{"note_id": note_id, "quote": quote}],
            "reasoning": "Documented.",
        }
        call = ToolCall(id=f"toolu_{criterion_id}", name="submit_finding", input=finding)
        return ModelResponse(tool_calls=[call], stop_reason=StopReason.TOOL_USE, usage=Usage(input_tokens=500))


def test_the_multi_agent_pipeline_records_in_its_own_folder_and_replays(gold, tmp_path):
    recordings = tmp_path / "rec"
    common = {"gold": gold, "recordings": recordings, "trials": 1, "pipeline": Pipeline.MULTI_AGENT}
    recorded = run_split("dev", Mode.RECORD, POLICIES, client=EvidenceModel(), **common)
    assert all(r.record.pipeline is Pipeline.MULTI_AGENT and all(c.passed for c in r.checks) for r in recorded)
    assert recording_path(recordings / "multi_agent", "dev", "dev_900", 1).exists()
    assert not (recordings / "dev").exists()  # the single agent's recordings are untouched
    replayed = run_split("dev", Mode.REPLAY, POLICIES, **common)
    assert [r.record.determination for r in replayed] == [r.record.determination for r in recorded]
    single = run_split("dev", Mode.REPLAY, POLICIES, gold=gold, recordings=recordings, trials=1)
    assert all(r.error.startswith("no recording at") for r in single)  # each pipeline replays only its own


def test_command_line_chooses_the_pipeline(gold, tmp_path, monkeypatch, capsys):
    monkeypatch.setattr("evals.runner.load_policies", lambda directory: POLICIES)
    common = {"gold": gold, "recordings": tmp_path / "rec", "trials": 1, "pipeline": Pipeline.MULTI_AGENT}
    run_split("dev", Mode.RECORD, POLICIES, client=EvidenceModel(), **common)
    args = ["replay", "--gold", str(gold), "--recordings", str(tmp_path / "rec"), "--trials", "1"]
    assert main([*args, "--pipeline", "multi_agent"]) == 0
    assert main(args) == 1  # the single agent has no recordings here
    assert "dev_900 trial 1: ERROR no recording" in capsys.readouterr().out


def test_the_command_line_records_and_replays_each_provider_in_its_own_folder(gold, tmp_path, monkeypatch, capsys):
    built = []
    monkeypatch.setattr("evals.runner.load_policies", lambda directory: POLICIES)
    monkeypatch.setattr(providers, "make_client", lambda p: built.append(p) or ScriptedClient(live_replies(2)))
    common = ["--gold", str(gold), "--recordings", str(tmp_path / "rec"), "--trials", "1"]
    assert main(["record", "--provider", "bedrock", *common]) == 0
    assert built == [Provider.BEDROCK]
    recorded = [p.relative_to(tmp_path / "rec") for p in (tmp_path / "rec").rglob("*.json")]
    assert recorded and all(p.parts[0] == "bedrock" for p in recorded)
    assert main(["replay", "--provider", "bedrock", *common]) == 0
    assert main(["replay", *common]) == 1  # nothing recorded on the Claude API
    assert "ERROR no recording" in capsys.readouterr().out
