"""Log an evaluation run to MLflow: what was run, how it scored, and the summary as an artifact.

Tracking is local: runs go to a SQLite file (mlflow.db) with artifacts under mlruns/, both gitignored.
CI does not log. A live evaluation is logged on the machine that ran it, and its summary is committed.

    python -m evals.runner replay --track
    mlflow ui --backend-store-uri sqlite:///mlflow.db     # the UI needs the full mlflow package
"""

import hashlib
import json

from evals.gold import GoldCase
from evals.runner import TrialResult, summary_line
from evals.scoring import Summary
from priorauth.policies import fingerprint
from priorauth.schemas import Policy

EXPERIMENT = "priorauth-eval"
TRACKING_URI = "sqlite:///mlflow.db"


def gold_fingerprint(cases: dict[str, GoldCase]) -> str:
    """Short hash of every case scored, so each run records exactly which gold set it was measured on."""
    canonical = json.dumps([cases[case_id].model_dump(mode="json") for case_id in sorted(cases)], sort_keys=True)
    return hashlib.sha256(canonical.encode()).hexdigest()[:16]


def run_params(
    results: list[TrialResult], cases: dict[str, GoldCase], policies: dict[str, Policy], split: str, mode: str
) -> dict[str, str]:
    """What was run: the split, the model and prompt, and fingerprints of the policies and gold cases."""
    records = [r.record for r in results if r.record]
    scored = {r.case_id: cases[r.case_id] for r in results}

    def joined(values: set[str]) -> str:
        return ",".join(sorted(values)) or "none"

    return {
        "split": split,
        "mode": mode,
        "cases": str(len(scored)),
        "trials": str(len(results)),
        "pipeline": joined({r.pipeline.value for r in records}),
        "model": joined({r.model for r in records}),
        "prompt_version": joined({r.prompt_version for r in records}),
        "prompt_fingerprint": joined({r.prompt_fingerprint for r in records}),
        "policy_fingerprints": joined({fingerprint(policies[case.policy_id]) for case in scored.values()}),
        "gold_fingerprint": gold_fingerprint(scored),
    }


def run_metrics(summary: Summary) -> dict[str, float]:
    """How it scored: every number in the summary, with one pass rate per check."""
    metrics = {
        "decision_accuracy": summary.decision_accuracy,
        "criterion_accuracy": summary.criterion_accuracy,
        "hard_gate_failures": float(len(summary.hard_gate_failures)),
        "errors": float(summary.errors),
        "unstable_cases": float(len(summary.unstable_cases)),
        "input_tokens": float(summary.input_tokens),
        "output_tokens": float(summary.output_tokens),
    }
    metrics |= {f"pass_rate.{name}": rate for name, rate in summary.check_pass_rates.items()}
    if summary.cost_usd is not None:
        metrics["cost_usd"] = summary.cost_usd
    return metrics


def log_run(
    summary: Summary,
    results: list[TrialResult],
    cases: dict[str, GoldCase],
    policies: dict[str, Policy],
    *,
    split: str,
    mode: str,
    tracking_uri: str = TRACKING_URI,
) -> str:
    """Log one evaluation run and return its MLflow run ID."""
    import mlflow  # optional dependency: the track extra

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(EXPERIMENT)
    params = run_params(results, cases, policies, split, mode)
    with mlflow.start_run(run_name=f"{split}-{mode}-{params['pipeline']}") as run:
        mlflow.log_params(params)
        mlflow.log_metrics(run_metrics(summary))
        mlflow.log_dict(summary.model_dump(mode="json"), "summary.json")
        mlflow.log_text("\n".join(summary_line(result) for result in results) + "\n", "trials.txt")
    return run.info.run_id
