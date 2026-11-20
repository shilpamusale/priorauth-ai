"""Score retrieval: does each request reach the right policy, or escalate for the right reason?

The cases are the gold cases of a split, each of which must reach its own policy, plus the routing-only
cases in evals/gold/routing/<split>.yaml: no LCD close enough, an LCD with no policy here, or a code its
policies do not cover. No model is called. Held-out cases are refused unless the final evaluation asks.
The fresh set is routing-only: requests written after the held-out run, the test for any change to retrieval.

The hard gate is that no request reaches a policy it should not. A request that escalates when it could
have matched is a safe miss: a person sees it, and it counts against accuracy.

    python -m evals.routing --min-accuracy 0.95
    python -m evals.routing --split fresh
    python -m evals.routing --split fresh --compare     # embedding-only and hybrid retrieval, side by side
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
from priorauth.retrieval import MIN_KEYWORD, Outcome, PolicyIndex, Retrieval, Route, lcd_name, route
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


def run(
    cases: list[RoutingCase],
    index: PolicyIndex,
    policies: dict[str, Policy],
    retrieval: Retrieval = Retrieval.EMBEDDING,
) -> list[RoutingResult]:
    return [
        RoutingResult(
            case=c, route=route(index, policies.values(), c.ordered_service, c.service_code, retrieval=retrieval)
        )
        for c in cases
    ]


def label(result: RoutingResult) -> str:
    return result.route.policy_id or result.route.outcome.value


def report(results: list[RoutingResult]) -> str:
    """One line per case, with the closest LCD and its distance, then the total."""
    lines = []
    for r in results:
        best = r.route.candidates[0] if r.route.candidates else None
        closest = f"{lcd_name(best.source)} at {best.distance:.3f}" if best else "-"
        want = r.case.policy_id or r.case.expected.value
        mark = "ok    " if r.correct else "UNSAFE" if r.unsafe else "WRONG "
        by_keyword = (
            f" (reached by keyword, {max(c.keyword for c in r.route.candidates):.2f})" if r.route.by_keyword else ""
        )
        lines.append(f"{mark} {r.case.case_id:<18} want {want:<18} got {label(r):<18} closest {closest}{by_keyword}")
    correct = sum(r.correct for r in results)
    lines.append(f"Routed correctly: {correct} of {len(results)} ({correct / len(results):.0%})")
    lines.append(f"Unsafe routes (reached a policy they should not): {sum(r.unsafe for r in results)}")
    return "\n".join(lines)


def compare_report(embedding: list[RoutingResult], hybrid: list[RoutingResult]) -> str:
    """Both modes' totals, every route hybrid changes, and the keyword score of each order the embeddings reject."""
    n = len(embedding)
    correct = [sum(r.correct for r in rs) for rs in (embedding, hybrid)]
    unsafe = [sum(r.unsafe for r in rs) for rs in (embedding, hybrid)]
    lines = [
        f"{'':<20}{'embedding':>12}{'hybrid':>12}",
        f"{'Routed correctly':<20}{f'{correct[0]} of {n}':>12}{f'{correct[1]} of {n}':>12}",
        f"{'Unsafe routes':<20}{unsafe[0]:>12}{unsafe[1]:>12}",
    ]
    changed = [(e, h) for e, h in zip(embedding, hybrid, strict=True) if label(e) != label(h)]
    lines.append(f"Routes hybrid changes: {len(changed)}")
    for e, h in changed:
        mark = "fixed " if h.correct else "UNSAFE" if h.unsafe else "broke " if e.correct else "wrong "
        want = e.case.policy_id or e.case.expected.value
        lines.append(f"  {mark} {e.case.case_id:<18} want {want:<18} embedding {label(e):<18} hybrid {label(h)}")
    rejected = [h for e, h in zip(embedding, hybrid, strict=True) if e.route.outcome is Outcome.NO_MATCH]
    lines.append(f"Orders the embeddings find too far, with their best keyword match (minimum {MIN_KEYWORD}):")
    for h in rejected:
        words = max(h.route.candidates, key=lambda c: c.keyword)
        lines.append(f"  {h.case.case_id:<18} {lcd_name(words.source)} {words.keyword:.2f}  {h.case.ordered_service}")
    return "\n".join(lines)


def main(argv: list[str] | None = None, make_index: Callable[[dict[str, str]], PolicyIndex] = PolicyIndex) -> int:
    parser = argparse.ArgumentParser(description="Score policy retrieval on the gold and routing cases.")
    parser.add_argument("--split", default="dev", choices=SETS)
    parser.add_argument("--allow-heldout", action="store_true", help="only for the final evaluation")
    parser.add_argument("--gold", type=Path, default=GOLD)
    parser.add_argument("--policies", type=Path, default=POLICIES)
    parser.add_argument("--min-accuracy", type=float, default=0.0, help="CI gate threshold")
    parser.add_argument("--retrieval", default=Retrieval.EMBEDDING.value, choices=[r.value for r in Retrieval])
    parser.add_argument("--compare", action="store_true", help="run both retrieval modes and compare them")
    args = parser.parse_args(argv)

    policies = load_policies(args.policies)
    cases = routing_cases(args.gold, args.split, policies, allow_heldout=args.allow_heldout)
    index = make_index(load_sources(args.policies))
    if args.compare:
        embedding, hybrid = (run(cases, index, policies, retrieval) for retrieval in Retrieval)
        print(compare_report(embedding, hybrid))
        return 1 if any(r.unsafe for r in embedding + hybrid) else 0
    results = run(cases, index, policies, Retrieval(args.retrieval))
    print(report(results))
    accuracy = sum(r.correct for r in results) / len(results)
    return 1 if any(r.unsafe for r in results) or accuracy < args.min_accuracy else 0


if __name__ == "__main__":
    raise SystemExit(main())
