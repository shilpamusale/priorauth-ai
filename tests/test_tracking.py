"""MLflow tracking: what a run logs, checked against a local SQLite store in a temporary folder."""

from pathlib import Path

import mlflow
import pytest

from evals.gold import load_split
from evals.runner import Mode, run_split
from evals.scoring import summarize
from evals.tracking import EXPERIMENT, gold_fingerprint, log_run, run_metrics, run_params
from priorauth.policies import fingerprint, load_policies


@pytest.fixture(scope="module")
def replayed():
    """The committed dev recordings, replayed and scored, as CI does."""
    policies = load_policies(Path("policies"))
    cases = load_split(Path("evals/gold"), "dev", policies)
    results = run_split("dev", Mode.REPLAY, policies, trials=1, cases=["dev_001", "dev_003"])
    return policies, cases, results, summarize(results, cases)


def test_params_record_what_was_run(replayed):
    policies, cases, results, _ = replayed
    params = run_params(results, cases, policies, "dev", "replay")
    assert params["split"] == "dev" and params["mode"] == "replay"
    assert (params["cases"], params["trials"]) == ("2", "2")
    assert params["model"] == "claude-sonnet-5-5" and params["prompt_version"] == "matcher_v1"
    assert params["pipeline"] == "single_agent"
    assert params["policy_fingerprints"] == fingerprint(policies["l38773_first_mbb"])
    assert params["gold_fingerprint"] == gold_fingerprint({c: cases[c] for c in ["dev_001", "dev_003"]})


def test_gold_fingerprint_changes_with_any_case_and_ignores_order(replayed):
    _, cases, _, _ = replayed
    two = {"dev_001": cases["dev_001"], "dev_003": cases["dev_003"]}
    assert gold_fingerprint(two) == gold_fingerprint({"dev_003": cases["dev_003"], "dev_001": cases["dev_001"]})
    relabelled = cases["dev_003"].model_copy(update={"tags": ["edited"]})
    assert gold_fingerprint(two) != gold_fingerprint({**two, "dev_003": relabelled})


def test_metrics_carry_every_number_in_the_summary(replayed):
    *_, summary = replayed
    metrics = run_metrics(summary)
    assert metrics["decision_accuracy"] == summary.decision_accuracy
    assert metrics["hard_gate_failures"] == 0.0 and metrics["errors"] == 0.0
    assert metrics["pass_rate.evidence_grounded"] == 1.0
    assert len([k for k in metrics if k.startswith("pass_rate.")]) == 6
    assert metrics["cost_usd"] == summary.cost_usd and metrics["input_tokens"] > 0


def test_unknown_cost_is_left_out_rather_than_logged_as_zero(replayed):
    *_, summary = replayed
    assert "cost_usd" not in run_metrics(summary.model_copy(update={"cost_usd": None}))


def test_log_run_writes_params_metrics_and_artifacts(replayed, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)  # MLflow puts artifacts under ./mlruns
    policies, cases, results, summary = replayed
    uri = f"sqlite:///{tmp_path / 'mlflow.db'}"
    run_id = log_run(summary, results, cases, policies, split="dev", mode="replay", tracking_uri=uri)
    client = mlflow.tracking.MlflowClient(tracking_uri=uri)
    run = client.get_run(run_id)
    assert client.get_experiment(run.info.experiment_id).name == EXPERIMENT
    assert run.info.run_name == "dev-replay-single_agent"
    assert run.data.params == run_params(results, cases, policies, "dev", "replay")
    assert run.data.metrics == run_metrics(summary)
    assert sorted(a.path for a in client.list_artifacts(run_id)) == ["summary.json", "trials.txt"]
    trials = Path(client.download_artifacts(run_id, "trials.txt", str(tmp_path))).read_text()
    assert trials.splitlines()[0] == "dev_001 trial 1: ready_to_submit, all checks passed"


def test_command_line_logs_the_run_when_asked(tmp_path, monkeypatch, capsys):
    from evals.runner import main

    repo = Path.cwd()
    monkeypatch.chdir(tmp_path)  # mlflow.db and mlruns/ are written to the current folder
    paths = ["--gold", str(repo / "evals/gold"), "--recordings", str(repo / "evals/recordings")]
    args = ["replay", "--cases", "dev_001", "--trials", "1", "--policies", str(repo / "policies"), *paths]
    assert main(args) == 0
    assert not (tmp_path / "mlflow.db").exists()  # no tracking unless asked
    assert main([*args, "--track"]) == 0
    assert "MLflow run " in capsys.readouterr().out and (tmp_path / "mlflow.db").exists()
