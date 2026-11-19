"""Compare the same evaluation run on two providers: the Claude API and Amazon Bedrock.

Both sets of recordings are replayed through the same code and scored against the same gold labels.
The report gives each provider's scores side by side, then where they differ: a case whose decision
differs in any trial, and each criterion whose status differs in the same case and trial. The model is
not deterministic, so the providers can differ the way two runs on one provider do; the report counts
differences rather than declaring one provider right.

    python -m evals.providers                          # dev cases
    python -m evals.providers --split heldout --allow-heldout
"""

import argparse
from pathlib import Path

from evals.gold import SPLITS, load_split
from evals.runner import GOLD, POLICIES, RECORDINGS, TRIALS, Mode, TrialResult, provider_root, run_split
from evals.scoring import Summary, summarize
from priorauth.llm.providers import Provider
from priorauth.policies import load_policies
from priorauth.schemas import Contract


class Comparison(Contract):
    summaries: dict[Provider, Summary]
    decision_differences: list[str]  # "case trial N: anthropic X, bedrock Y"
    criterion_differences: list[str]  # "case trial N criterion: anthropic X, bedrock Y"
    compared_criteria: int


def compare(results: dict[Provider, list[TrialResult]], cases) -> Comparison:
    """Scores per provider, and every decision and criterion status that differs between them."""
    a, b = Provider.ANTHROPIC, Provider.BEDROCK
    by_run = {p: {(r.case_id, r.trial): r for r in rs if r.record} for p, rs in results.items()}
    decisions, criteria, compared = [], [], 0
    for key in sorted(by_run[a].keys() & by_run[b].keys()):
        left, right = by_run[a][key].record.determination, by_run[b][key].record.determination
        where = f"{key[0]} trial {key[1]}"
        if left.decision is not right.decision:
            decisions.append(f"{where}: {a} {left.decision}, {b} {right.decision}")
        left_status = {r.criterion_id: r.status for r in left.criteria}
        right_status = {r.criterion_id: r.status for r in right.criteria}
        for cid in sorted(left_status.keys() | right_status.keys()):
            compared += 1
            left_one, right_one = left_status.get(cid, "no finding"), right_status.get(cid, "no finding")
            if left_one != right_one:
                criteria.append(f"{where} {cid}: {a} {left_one}, {b} {right_one}")
    return Comparison(
        summaries={p: summarize(rs, cases) for p, rs in results.items()},
        decision_differences=decisions,
        criterion_differences=criteria,
        compared_criteria=compared,
    )


def report(comparison: Comparison) -> str:
    rows = [("runs", lambda s: f"{s.trials - s.errors}"), ("decision accuracy", lambda s: f"{s.decision_accuracy:.2f}")]
    rows += [
        ("criterion accuracy", lambda s: f"{s.criterion_accuracy:.3f}"),
        ("hard-gate failures", lambda s: f"{len(s.hard_gate_failures)}"),
        ("unstable cases", lambda s: f"{len(s.unstable_cases)}"),
        ("cost", lambda s: "unknown" if s.cost_usd is None else f"${s.cost_usd:.2f}"),
    ]
    providers = list(comparison.summaries)
    lines = [f"{'':<20}" + "".join(f"{p.value:>12}" for p in providers)]
    for name, cell in rows:
        values = [cell(s) if s.trials > s.errors or name == "runs" else "-" for s in comparison.summaries.values()]
        lines.append(f"{name:<20}" + "".join(f"{value:>12}" for value in values))
    for provider, summary in comparison.summaries.items():
        if summary.errors:
            lines.append(f"{provider}: {summary.errors} of {summary.trials} runs have no usable recording")
    lines.append(f"Decisions that differ between providers: {len(comparison.decision_differences)}")
    lines += [f"  {line}" for line in comparison.decision_differences]
    differing = len(comparison.criterion_differences)
    lines.append(f"Criterion statuses that differ: {differing} of {comparison.compared_criteria}")
    lines += [f"  {line}" for line in comparison.criterion_differences]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Compare the Claude API and Bedrock recordings of one split.")
    parser.add_argument("--split", default="dev", choices=SPLITS)
    parser.add_argument("--allow-heldout", action="store_true")
    parser.add_argument("--trials", type=int, default=TRIALS)
    parser.add_argument("--gold", type=Path, default=GOLD)
    parser.add_argument("--recordings", type=Path, default=RECORDINGS)
    parser.add_argument("--policies", type=Path, default=POLICIES)
    args = parser.parse_args(argv)

    policies = load_policies(args.policies)
    results = {
        provider: run_split(
            args.split,
            Mode.REPLAY,
            policies,
            gold=args.gold,
            recordings=provider_root(args.recordings, provider),
            trials=args.trials,
            allow_heldout=args.allow_heldout,
        )
        for provider in (Provider.ANTHROPIC, Provider.BEDROCK)
    }
    comparison = compare(results, load_split(args.gold, args.split, policies))
    print(report(comparison))
    return 1 if any(s.errors for s in comparison.summaries.values()) else 0


if __name__ == "__main__":
    raise SystemExit(main())
