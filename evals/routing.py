"""Score retrieval: does each request reach the right policy, or escalate for the right reason?

The cases are the gold cases of a split, each of which must reach its own policy, plus the routing-only
cases in evals/gold/routing/<split>.yaml: no LCD close enough, an LCD with no policy here, or a code its
policies do not cover. No model is called. Held-out cases are refused unless the final evaluation asks.
The fresh set is routing-only: requests written after the held-out run, the test for any change to retrieval.

The hard gate is that no request reaches a policy it should not. A request that escalates when it could
have matched is a safe miss: a person sees it, and it counts against accuracy.

    python -m evals.routing --min-accuracy 0.95
    python -m evals.routing --split fresh
"""

import argparse
from collections.abc import Callable
from pathlib import Path
from typing import Self

import yaml
from pydantic import model_validator

from evals.gold import load_split
from evals.runner import GOLD, POLICIES
from priorauth.policies import load_policies, load_sources
from priorauth.retrieval import Outcome, PolicyIndex, Route, lcd_name, route
from priorauth.schemas import Contract, NonBlank, Policy, ServiceCode

SETS = ("dev", "heldout", "fresh")  # fresh has no gold cases, only routing-only requests


class RoutingCase(Contract):
    case_id: str
    ordered_service: NonBlank
    service_code: ServiceCode
    expected: Outcome
    policy_id: str | None = None  # the policy a matched request must reach

    @model_validator(mode="after")
    def _policy_only_when_matched(self) -> Self:
        if (self.expected is Outcome.MATCHED) != (self.policy_id is not None):
            raise ValueError(f"{self.case_id}: a policy_id is given exactly when the expected outcome is matched")
        return self


class RoutingResult(Contract):
    case: RoutingCase
    route: Route

    @property
    def correct(self) -> bool:
        return self.route.outcome is self.case.expected and self.route.policy_id == self.case.policy_id

    @property
    def unsafe(self) -> bool:
        """Sent to a policy it should not reach, where a wrong escalation would only cost a person's time."""
        return self.route.outcome is Outcome.MATCHED and not self.correct


def routing_cases(
    gold: Path, split: str, policies: dict[str, Policy], allow_heldout: bool = False
) -> list[RoutingCase]:
    """The split's gold cases as matched routes, then its routing-only cases."""
    if split not in SETS:
        raise ValueError(f"unknown routing set {split!r}; use one of {SETS}")
    if split == "heldout" and not allow_heldout:
        raise ValueError("held-out cases run only in the final evaluation; pass allow_heldout=True for that run")
    gold_cases = {} if split == "fresh" else load_split(gold, split, policies)
    cases = [
        RoutingCase(
            case_id=case.case_id,
            ordered_service=case.request.ordered_service,
            service_code=case.request.service_code,
            expected=Outcome.MATCHED,
            policy_id=case.policy_id,
        )
        for case in gold_cases.values()
    ]
    for data in yaml.safe_load((gold / "routing" / f"{split}.yaml").read_text(encoding="utf-8")):
        case = RoutingCase.model_validate(data)
        if case.policy_id is not None and case.policy_id not in policies:
            raise ValueError(f"{case.case_id}: unknown policy {case.policy_id!r}")
        cases.append(case)
    if len({c.case_id for c in cases}) != len(cases):
        raise ValueError(f"case IDs repeat in the {split} routing cases")
    return cases


def run(cases: list[RoutingCase], index: PolicyIndex, policies: dict[str, Policy]) -> list[RoutingResult]:
    return [
        RoutingResult(case=c, route=route(index, policies.values(), c.ordered_service, c.service_code)) for c in cases
    ]


def report(results: list[RoutingResult]) -> str:
    """One line per case, with the closest LCD and its distance, then the total."""
    lines = []
    for r in results:
        best = r.route.candidates[0] if r.route.candidates else None
        closest = f"{lcd_name(best.source)} at {best.distance:.3f}" if best else "-"
        got = r.route.policy_id or r.route.outcome.value
        want = r.case.policy_id or r.case.expected.value
        mark = "ok    " if r.correct else "UNSAFE" if r.unsafe else "WRONG "
        lines.append(f"{mark} {r.case.case_id:<18} want {want:<18} got {got:<18} closest {closest}")
    correct = sum(r.correct for r in results)
    lines.append(f"Routed correctly: {correct} of {len(results)} ({correct / len(results):.0%})")
    lines.append(f"Unsafe routes (reached a policy they should not): {sum(r.unsafe for r in results)}")
    return "\n".join(lines)


def main(argv: list[str] | None = None, make_index: Callable[[dict[str, str]], PolicyIndex] = PolicyIndex) -> int:
    parser = argparse.ArgumentParser(description="Score policy retrieval on the gold and routing cases.")
    parser.add_argument("--split", default="dev", choices=SETS)
    parser.add_argument("--allow-heldout", action="store_true", help="only for the final evaluation")
    parser.add_argument("--gold", type=Path, default=GOLD)
    parser.add_argument("--policies", type=Path, default=POLICIES)
    parser.add_argument("--min-accuracy", type=float, default=0.0, help="CI gate threshold")
    args = parser.parse_args(argv)

    policies = load_policies(args.policies)
    cases = routing_cases(args.gold, args.split, policies, allow_heldout=args.allow_heldout)
    results = run(cases, make_index(load_sources(args.policies)), policies)
    print(report(results))
    accuracy = sum(r.correct for r in results) / len(results)
    return 1 if any(r.unsafe for r in results) or accuracy < args.min_accuracy else 0


if __name__ == "__main__":
    raise SystemExit(main())
