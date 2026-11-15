"""Calibration of the LLM judge against the blind hand labels.

These tests use the committed items, sample and labels with made-up judge grades, so they never depend on
the judge's recordings: the judge is not a merge gate. CI runs the real calibration in its own step.
"""

import json

import pytest

from evals import calibration
from evals.calibration import agreement, calibrate, cohen_kappa, describe, main
from evals.judge import Grade, Judgement
from evals.judge_items import SAMPLE, HumanLabel, ItemKind, build_items, load_labels, load_sample
from evals.runner import GOLD, POLICIES, RECORDINGS
from priorauth.policies import load_policies


def test_kappa_matches_a_hand_calculation():
    # 10 items, 7 agreed. Rater one says "yes" 5 times, rater two 6 times, so chance agreement is
    # (5*6 + 5*4) / 100 = 0.5 and kappa is (0.7 - 0.5) / (1 - 0.5) = 0.4.
    pairs = [("yes", "yes")] * 4 + [("no", "no")] * 3 + [("yes", "no")] + [("no", "yes")] * 2
    assert cohen_kappa(pairs) == pytest.approx(0.4)


def test_kappa_is_one_for_perfect_agreement_zero_for_chance_and_undefined_with_one_grade():
    assert cohen_kappa([("a", "a"), ("b", "b")]) == 1.0
    assert cohen_kappa([("a", "a"), ("a", "b"), ("b", "a"), ("b", "b")]) == 0.0
    assert cohen_kappa([("a", "a"), ("a", "a")]) is None
    assert agreement([("a", "a"), ("a", "a")]) == calibration.Agreement(items=2, agreed=2, kappa=None)


@pytest.fixture(scope="module")
def committed():
    items, _ = build_items(load_policies(POLICIES), GOLD, RECORDINGS)
    return {i.item_id: i for i in items}, load_sample(SAMPLE), load_labels()


def judge_as(labels, **changes):
    """Judge grades equal to the person's labels, except the given item IDs."""
    grades = {i: label.label for i, label in labels.items()} | changes
    return {i: Judgement(item_id=i, grade=Grade(item_id=i, grade=g, reasoning=f"why {g}")) for i, g in grades.items()}


def first(items, sample, kind):
    return next(i for i in sample if items[i].kind is kind)


def test_the_sample_splits_into_evidence_planted_and_requests(committed):
    items, sample, labels = committed
    report = calibrate(items, sample, labels, judge_as(labels))
    assert (report.evidence.items, report.planted, report.requests.items) == (32, 8, 8)
    assert report.evidence.agreed == 32 and report.requests.agreed == 8 and report.disagreements == []


def test_a_disagreement_carries_both_sides_reasons(committed):
    items, sample, labels = committed
    item = first(items, sample, ItemKind.EVIDENCE)
    person = labels | {item: HumanLabel(label="partly", note="Timing of the baseline is not shown.")}
    report = calibrate(items, sample, person, judge_as(labels, **{item: "supports"}))
    [only] = report.disagreements
    assert (only.item_id, only.person, only.judge) == (item, "partly", "supports")
    assert (only.person_note, only.judge_reasoning) == ("Timing of the baseline is not shown.", "why supports")
    assert report.evidence.agreed == 31 and report.evidence.kappa < 1


def test_supports_or_not_ignores_partly_against_does_not_support(committed):
    items, sample, labels = committed
    planted = first(items, sample, ItemKind.PLANTED)
    other = "partly" if labels[planted].label == "does_not_support" else "does_not_support"
    report = calibrate(items, sample, labels, judge_as(labels, **{planted: other}))
    assert report.evidence.agreed == 31 and report.evidence_supports_or_not.agreed == 32


def test_a_planted_item_is_caught_by_any_grade_but_supports(committed):
    items, sample, labels = committed
    planted = [i for i in sample if items[i].kind is ItemKind.PLANTED]
    person = labels | {planted[0]: HumanLabel(label="supports"), planted[4]: HumanLabel(label="partly")}
    judge = judge_as(labels, **{planted[1]: "supports", planted[2]: "partly", planted[3]: "does_not_support"})
    report = calibrate(items, sample, person, judge)
    assert report.planted_caught_by_person == report.planted_caught_by_judge == 7  # partly counts as caught


def test_judge_grade_counts_cover_every_evidence_item_and_only_evidence(committed):
    items, sample, labels = committed
    evidence = [i for i, item in items.items() if item.kind is ItemKind.EVIDENCE]
    judgements = judge_as(labels) | {
        i: Judgement(item_id=i, grade=Grade(item_id=i, grade="partly", reasoning="why partly")) for i in evidence
    }
    report = calibrate(items, sample, labels, judgements)
    assert report.judge_grades_all_evidence == {"partly": len(evidence)}


def test_a_missing_label_or_grade_is_an_error(committed):
    items, sample, labels = committed
    with pytest.raises(ValueError, match=sample[0]):
        calibrate(items, sample, {k: v for k, v in labels.items() if k != sample[0]}, judge_as(labels))
    errored = judge_as(labels) | {sample[1]: Judgement(item_id=sample[1], error="the run differs from the recording")}
    with pytest.raises(ValueError, match=sample[1]):
        calibrate(items, sample, labels, errored)


def test_command_line_prints_and_writes_the_report(committed, monkeypatch, tmp_path, capsys):
    items, sample, labels = committed
    item = first(items, sample, ItemKind.REQUEST)
    judgements = judge_as(labels, **{item: "needs_revision"})
    monkeypatch.setattr(calibration, "load_all", lambda: (items, {}, sample, judgements))
    out = tmp_path / "calibration.json"
    assert main(["--out", str(out)]) == 0
    printed = capsys.readouterr().out
    assert printed == describe(calibrate(items, sample, labels, judgements)) + "\n"
    assert "Requests: agreed on 7 of 8 (88%)" in printed and "Disagreements (1):" in printed
    assert json.loads(out.read_text())["requests"] == {"items": 8, "agreed": 7, "kappa": 0.0}
