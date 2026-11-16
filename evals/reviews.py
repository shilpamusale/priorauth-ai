"""The review queue, saved reviews, and the candidate gold cases they become.

The queue holds every dev case the single agent escalated or whose trials disagree (priorauth.review).
A reviewer works blind to the gold label, on the local screen ui/review.py, and each review is saved as
evals/reviews/<case_id>.yaml. This command then turns each review into a candidate gold case in
evals/gold/candidates/, for a person to check and promote, and reports where a review disagrees with the
existing gold label, which is how a doubtful label gets a second look.

    python -m evals.reviews             # the queue, then the reviews: candidates written, disagreements listed
"""

import argparse
from pathlib import Path

import yaml

from evals.gold import GoldCase, load_split
from evals.runner import GOLD, POLICIES, RECORDINGS, Mode, run_split
from priorauth.policies import load_policies
from priorauth.review import Review, review_reasons, reviewed_determination
from priorauth.schemas import Contract, Determination, Policy

SPLIT = "dev"
REVIEWS = Path("evals/reviews")
CANDIDATES = Path("evals/gold/candidates")
REVIEWED_TRIAL = 1  # the trial the reviewer corrects; the other trials show where the model wavered


class QueueItem(Contract):
    case_id: str
    trials: dict[int, Determination]
    reasons: list[str]


def recorded_trials(
    policies: dict[str, Policy], gold: Path = GOLD, recordings: Path = RECORDINGS
) -> dict[str, dict[int, Determination]]:
    """Each dev case's recorded single-agent determinations, by trial."""
    by_case: dict[str, dict[int, Determination]] = {}
    for result in run_split(SPLIT, Mode.REPLAY, policies, gold=gold, recordings=recordings):
        if result.error:
            raise ValueError(f"{result.case_id} trial {result.trial}: {result.error}")
        by_case.setdefault(result.case_id, {})[result.trial] = result.record.determination
    return by_case


def review_queue(trials: dict[str, dict[int, Determination]]) -> list[QueueItem]:
    """Every case whose trials need a person, with the reasons, in case order."""
    items = [QueueItem(case_id=c, trials=t, reasons=review_reasons(t)) for c, t in trials.items()]
    return [item for item in items if item.reasons]


def review_path(reviews: Path, case_id: str) -> Path:
    return reviews / f"{case_id}.yaml"


def save_review(review: Review, reviews: Path = REVIEWS) -> Path:
    path = review_path(reviews, review.case_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(review.model_dump(mode="json"), sort_keys=False), encoding="utf-8")
    return path


def load_reviews(reviews: Path = REVIEWS) -> dict[str, Review]:
    loaded = {}
    for path in sorted(reviews.glob("*.yaml")):
        review = Review.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))
        if review.case_id != path.stem:
            raise ValueError(f"{path.name}: case_id {review.case_id!r} must match the file name")
        loaded[review.case_id] = review
    return loaded


def candidate_case(case: GoldCase, policy: Policy, determination: Determination, review: Review) -> GoldCase:
    """The reviewed determination as a gold case: every criterion labelled, the decision from decide()."""
    found = {r.criterion_id: r for r in determination.criteria}
    if unlabelled := [c.criterion_id for c in policy.criteria if c.criterion_id not in found]:
        raise ValueError(f"{case.case_id}: the review leaves criteria with no status: {unlabelled}")
    labels = {
        cid: {"status": r.status, "evidence_notes": sorted({e.note_id for e in r.evidence})} for cid, r in found.items()
    }
    return GoldCase(
        policy_id=policy.policy_id,
        tags=["candidate", f"reviewed-by-{review.reviewer.lower().replace(' ', '-')}"],
        request=case.request,
        labels=labels,
        expected_decision=determination.decision,
    )


def gold_disagreements(case: GoldCase, candidate: GoldCase) -> list[str]:
    """Where the reviewed case differs from the existing gold label."""
    lines = [
        f"{cid}: gold {case.labels[cid].status}, review {label.status}"
        for cid, label in candidate.labels.items()
        if label.status is not case.labels[cid].status
    ]
    if candidate.expected_decision is not case.expected_decision:
        lines.append(f"decision: gold {case.expected_decision}, review {candidate.expected_decision}")
    return lines


def write_candidate(candidate: GoldCase, review: Review, candidates: Path = CANDIDATES) -> Path:
    path = candidates / f"{candidate.case_id}.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    header = (
        f"# Candidate gold case from {review.reviewer}'s review on {review.reviewed_at:%Y-%m-%d}.\n"
        "# Check it, then promote it by hand; candidates are never scored.\n"
    )
    data = candidate.model_dump(mode="json")
    data["labels"] = {
        cid: {k: v for k, v in label.items() if v or k == "status"} for cid, label in data["labels"].items()
    }
    path.write_text(header + yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    return path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="The review queue, and candidate gold cases from saved reviews.")
    parser.add_argument("--gold", type=Path, default=GOLD)
    parser.add_argument("--recordings", type=Path, default=RECORDINGS)
    parser.add_argument("--policies", type=Path, default=POLICIES)
    parser.add_argument("--reviews", type=Path, default=REVIEWS)
    parser.add_argument("--candidates", type=Path, default=CANDIDATES)
    args = parser.parse_args(argv)

    policies = load_policies(args.policies)
    trials = recorded_trials(policies, args.gold, args.recordings)
    queue = review_queue(trials)
    reviews = load_reviews(args.reviews)
    print(f"Review queue: {len(queue)} cases, {sum(item.case_id in reviews for item in queue)} reviewed")
    for item in queue:
        mark = "reviewed" if item.case_id in reviews else "to do   "
        print(f"  {mark} {item.case_id}: {'; '.join(item.reasons)}")

    cases = load_split(args.gold, SPLIT, policies)
    for case_id, review in reviews.items():
        case, policy = cases[case_id], policies[cases[case_id].policy_id]
        findings = trials[case_id][review.trial].criteria
        determination = reviewed_determination(policy, case.request, findings, review)
        candidate = candidate_case(case, policy, determination, review)
        path = write_candidate(candidate, review, args.candidates)
        differences = gold_disagreements(case, candidate)
        print(f"{case_id}: reviewed decision {determination.decision}; candidate written to {path}")
        for line in differences or ["agrees with the gold label"]:
            print(f"    {line}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
