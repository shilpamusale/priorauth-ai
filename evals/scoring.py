"""Metrics over a set of trials, and the pass/fail rule for the CI gate.

The gate fails on any hard-gate failure in any trial, on any trial that produced no record, and when
decision accuracy falls below the threshold. A good average never excuses a safety failure.
"""

from collections import defaultdict

from evals.gold import GoldCase
from evals.runner import TrialResult
from priorauth.pipeline import AuditRecord
from priorauth.schemas import Contract

PRICES = {  # US$ per million input and output tokens
    "claude-sonnet-5-5": (2.00, 10.00),
    "claude-opus-5-5": (4.00, 20.00),
}


class Summary(Contract):
    cases: int
    trials: int
    errors: int  # trials with no record: a missing or stale recording
    decision_accuracy: float  # share of trials with the expected decision
    criterion_accuracy: float  # share of criterion labels matched, over every trial
    check_pass_rates: dict[str, float]
    hard_gate_failures: list[str]  # one line per failure: case, trial, gate and detail
    unstable_cases: list[str]  # cases whose trials did not all reach the same decision
    input_tokens: int
    output_tokens: int
    cost_usd: float | None  # None when a model's price is unknown


def cost_usd(record: AuditRecord) -> float | None:
    if record.model not in PRICES:
        return None
    per_input, per_output = PRICES[record.model]
    return (record.usage.input_tokens * per_input + record.usage.output_tokens * per_output) / 1_000_000


def summarize(results: list[TrialResult], cases: dict[str, GoldCase]) -> Summary:
    """Rates are over the trials that produced a record; trials with errors are counted separately."""
    done = [r for r in results if r.record]

    def share(count: int, total: int) -> float:
        return count / total if total else 0.0

    labels = [
        (label.status, {c.criterion_id: c.status for c in r.record.determination.criteria}.get(cid))
        for r in done
        for cid, label in cases[r.case_id].labels.items()
    ]
    names = [c.name for c in done[0].checks] if done else []
    decisions = defaultdict(set)
    for r in done:
        decisions[r.case_id].add(r.record.determination.decision)
    costs = [cost_usd(r.record) for r in done]
    return Summary(
        cases=len({r.case_id for r in results}),
        trials=len(results),
        errors=len(results) - len(done),
        decision_accuracy=share(
            sum(r.record.determination.decision is cases[r.case_id].expected_decision for r in done), len(done)
        ),
        criterion_accuracy=share(sum(expected is got for expected, got in labels), len(labels)),
        check_pass_rates={
            name: share(sum(c.passed for r in done for c in r.checks if c.name == name), len(done)) for name in names
        },
        hard_gate_failures=[
            f"{r.case_id} trial {r.trial}: {c.name}: {c.detail}"
            for r in done
            for c in r.checks
            if c.hard_gate and not c.passed
        ],
        unstable_cases=sorted(case_id for case_id, found in decisions.items() if len(found) > 1),
        input_tokens=sum(r.record.usage.input_tokens for r in done),
        output_tokens=sum(r.record.usage.output_tokens for r in done),
        cost_usd=None if None in costs else round(sum(costs), 4),
    )


def gate(summary: Summary, min_decision_accuracy: float) -> list[str]:
    """Why the CI gate fails; an empty list means it passes."""
    failures = [f"hard gate: {failure}" for failure in summary.hard_gate_failures]
    if summary.errors:
        failures.append(f"{summary.errors} trials have no record")
    if summary.decision_accuracy < min_decision_accuracy:
        failures.append(f"decision accuracy {summary.decision_accuracy:.2f} is below {min_decision_accuracy:.2f}")
    return failures


def format_summary(summary: Summary) -> str:
    cost = "unknown" if summary.cost_usd is None else f"${summary.cost_usd:.2f}"
    rates = ", ".join(f"{name} {rate:.2f}" for name, rate in summary.check_pass_rates.items())
    return "\n".join(
        [
            f"{summary.cases} cases, {summary.trials} trials, {summary.errors} errors",
            f"decision accuracy {summary.decision_accuracy:.2f}, criterion accuracy {summary.criterion_accuracy:.2f}",
            f"checks: {rates or 'none'}",
            f"unstable cases: {', '.join(summary.unstable_cases) or 'none'}",
            f"hard-gate failures: {len(summary.hard_gate_failures)}",
            f"tokens: {summary.input_tokens:,} in, {summary.output_tokens:,} out, cost {cost}",
        ]
    )
