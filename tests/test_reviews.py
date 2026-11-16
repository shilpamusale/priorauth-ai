"""The review queue, saved reviews, and the candidate gold cases they become."""

import datetime
from pathlib import Path

import pytest

from evals.gold import load_case, load_split
from evals.reviews import (
    candidate_case,
    gold_disagreements,
    load_reviews,
    main,
    recorded_trials,
    review_queue,
    reviewed_determination,
    save_review,
    write_candidate,
)
from evals.runner import GOLD
from priorauth.policies import load_policies
from priorauth.review import Override, Review
from priorauth.schemas import CriterionStatus, Decision, Evidence

POLICIES = load_policies(Path("policies"))
CASES = load_split(GOLD, "dev", POLICIES)


@pytest.fixture(scope="module")
def trials():
    return recorded_trials(POLICIES)


def review(case_id="dev_005", **overrides):
    return Review(
        case_id=case_id,
        policy_id=CASES[case_id].policy_id,
        trial=1,
        reviewer="Ishi Musale",
        reviewed_at=datetime.datetime(2026, 11, 16, 9, 30, tzinfo=datetime.UTC),
        overrides=overrides,
    )


def test_the_queue_holds_the_escalations_and_the_cases_whose_trials_disagree(trials):
    queue = {item.case_id: item.reasons for item in review_queue(trials)}
    assert list(queue) == ["dev_005", "dev_006", "dev_009", "dev_010", "dev_011", "dev_012", "dev_017"]
    assert queue["dev_006"] == [
        "trials disagree on l38773_first_mbb.functional_deficit: "
        "trial 1 met, trial 2 met, trial 3 insufficient_evidence"
    ]
    assert queue["dev_009"][0] == "the agent escalated" and "no_other_pathology" in queue["dev_009"][1]


def test_a_saved_review_loads_back(tmp_path):
    saved = review(**{"l38773_first_mbb.duration": Override(status=CriterionStatus.NOT_MET, reason="Six weeks.")})
    path = save_review(saved, tmp_path)
    assert path == tmp_path / "dev_005.yaml" and load_reviews(tmp_path) == {"dev_005": saved}


def test_a_review_file_must_be_named_after_its_case(tmp_path):
    save_review(review(), tmp_path).rename(tmp_path / "dev_006.yaml")
    with pytest.raises(ValueError, match="must match the file name"):
        load_reviews(tmp_path)


def reviewed(trials, case_id, **overrides):
    case = CASES[case_id]
    policy = POLICIES[case.policy_id]
    given = review(case_id, **overrides)
    determination = reviewed_determination(policy, case.request, trials[case_id][1].criteria, given)
    return case, policy, determination, given


def test_a_review_that_confirms_the_agent_matches_the_gold_label(trials):
    case, policy, determination, given = reviewed(trials, "dev_005")
    candidate = candidate_case(case, policy, determination, given)
    assert candidate.expected_decision is Decision.ESCALATE and gold_disagreements(case, candidate) == []
    assert candidate.tags == ["candidate", "reviewed-by-ishi-musale"]


def test_a_review_that_differs_from_gold_says_where(trials):
    quote = Evidence(note_id="n4", quote="Axial low back pain for 6 weeks")
    met = Override(status=CriterionStatus.MET, reason="Counting from the first visit.", evidence=[quote])
    case, policy, determination, given = reviewed(trials, "dev_005", **{"l38773_first_mbb.duration": met})
    candidate = candidate_case(case, policy, determination, given)
    assert candidate.labels["l38773_first_mbb.duration"].evidence_notes == ["n4"]
    assert gold_disagreements(case, candidate) == [
        "l38773_first_mbb.duration: gold not_met, review met",
        f"decision: gold escalate, review {candidate.expected_decision}",
    ]


def test_every_criterion_needs_a_status_before_a_case_can_be_a_candidate(trials):
    case, policy, determination, given = reviewed(trials, "dev_005")
    partial = determination.model_copy(update={"criteria": determination.criteria[1:]})
    with pytest.raises(ValueError, match="no status"):
        candidate_case(case, policy, partial, given)


def test_a_written_candidate_is_a_valid_gold_case_and_is_never_scored(trials, tmp_path):
    case, policy, determination, given = reviewed(trials, "dev_005")
    path = write_candidate(candidate_case(case, policy, determination, given), given, tmp_path / "candidates")
    assert path.read_text().startswith("# Candidate gold case from Ishi Musale's review on 2026-11-16.")
    loaded = load_case(path, POLICIES)  # every gold rule holds: labels complete, decision follows from them
    assert {cid: label.status for cid, label in loaded.labels.items()} == {
        cid: label.status for cid, label in case.labels.items()
    }
    assert "candidates" not in {p.name for p in GOLD.iterdir() if p.is_dir() and p.name in ("dev", "heldout")}


def test_command_line_lists_the_queue_and_turns_reviews_into_candidates(tmp_path, capsys):
    reviews, candidates = tmp_path / "reviews", tmp_path / "candidates"
    save_review(review(), reviews)
    assert main(["--reviews", str(reviews), "--candidates", str(candidates)]) == 0
    printed = capsys.readouterr().out
    assert printed.startswith("Review queue: 7 cases, 1 reviewed\n  reviewed dev_005: the agent escalated\n")
    assert "  to do    dev_011: the agent escalated" in printed
    assert "dev_005: reviewed decision escalate; candidate written to" in printed
    assert "    agrees with the gold label" in printed and (candidates / "dev_005.yaml").exists()
