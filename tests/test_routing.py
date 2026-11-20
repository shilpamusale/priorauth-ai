"""The routing evaluation: every dev request must reach its policy or escalate for the stated reason.

These tests use a fake index with fixed distances; CI runs the real embedding model in its own step.
"""

import shutil
from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError
from test_retrieval import EYELID, FACET, VEINS, FakeIndex

from evals.routing import RoutingCase, RoutingResult, compare_report, main, report, routing_cases, run
from evals.runner import GOLD
from priorauth.policies import load_policies, load_sources
from priorauth.retrieval import MIN_KEYWORD, Candidate, KeywordIndex, Outcome, Retrieval, Route

POLICIES = load_policies(Path("policies"))
# where each routing-only dev case's order lands, and how close: the outcomes then follow from the code check
LANDS = {
    "route_dev_001": (VEINS, 0.2),
    "route_dev_002": (FACET, 0.3),
    "route_dev_003": (FACET, 0.3),
    "route_dev_004": (FACET, 0.9),
    "route_dev_005": (EYELID, 0.8),
    "route_dev_006": (EYELID, 0.2),
    "route_dev_007": (EYELID, 0.25),
}


class CaseIndex:
    """Sends each gold order to its policy's LCD, and routing-only orders where LANDS says."""

    def __init__(self, cases, lands=LANDS):
        home = {c.case_id: (POLICIES[c.policy_id].source, 0.25) for c in cases if c.policy_id}
        self.by_order = {c.ordered_service: lands.get(c.case_id, home.get(c.case_id, (FACET, 0.25))) for c in cases}

    def search(self, text):
        source, distance = self.by_order[text]
        return [Candidate(source=source, distance=distance)]


@pytest.fixture(scope="module")
def dev():
    return routing_cases(GOLD, "dev", POLICIES)


def test_dev_cases_are_the_gold_cases_then_the_routing_cases(dev):
    gold = sorted(path.stem for path in (GOLD / "dev").glob("*.yaml"))
    assert [c.case_id for c in dev[: len(gold)]] == gold
    assert [c.case_id for c in dev[len(gold) :]] == list(LANDS)
    assert {c.policy_id for c in dev if c.expected is Outcome.MATCHED} == set(POLICIES)  # every policy is reached


def test_held_out_cases_need_the_final_evaluation():
    with pytest.raises(ValueError, match="final evaluation"):
        routing_cases(GOLD, "heldout", POLICIES)
    gold = len(list((GOLD / "heldout").glob("*.yaml")))
    assert len(routing_cases(GOLD, "heldout", POLICIES, allow_heldout=True)) == gold + 3


def test_the_fresh_set_is_new_routing_only_requests_with_every_outcome():
    fresh = routing_cases(GOLD, "fresh", POLICIES)
    assert [c.case_id for c in fresh] == [f"fresh_{n:03d}" for n in range(1, 20)]
    outcomes = [c.expected for c in fresh]
    assert {o: outcomes.count(o) for o in Outcome} == {
        Outcome.MATCHED: 6,
        Outcome.NO_ORDER: 0,
        Outcome.NO_MATCH: 7,
        Outcome.UNSUPPORTED: 1,
        Outcome.CODE_NOT_COVERED: 5,
    }
    seen = routing_cases(GOLD, "dev", POLICIES) + routing_cases(GOLD, "heldout", POLICIES, allow_heldout=True)
    assert not {c.ordered_service for c in fresh} & {c.ordered_service for c in seen}  # nothing reused
    with pytest.raises(ValueError, match="unknown routing set"):
        routing_cases(GOLD, "test", POLICIES)


@pytest.mark.parametrize(
    "fields",
    [{"expected": "matched"}, {"expected": "no_match", "policy_id": "l38773_first_mbb"}],
)
def test_a_policy_is_named_exactly_when_the_route_should_match(fields):
    with pytest.raises(ValidationError, match="exactly when"):
        RoutingCase(case_id="c", ordered_service="o", service_code="64493", **fields)


def copy_gold(tmp_path, extra):
    gold = tmp_path / "gold"
    shutil.copytree(GOLD, gold)
    path = gold / "routing" / "dev.yaml"
    path.write_text(path.read_text() + yaml.safe_dump([extra]))
    return gold


def test_an_unknown_policy_in_a_routing_case_is_an_error(tmp_path):
    extra = {
        "case_id": "x",
        "ordered_service": "o",
        "service_code": "64493",
        "expected": "matched",
        "policy_id": "nope",
    }
    with pytest.raises(ValueError, match="unknown policy 'nope'"):
        routing_cases(copy_gold(tmp_path, extra), "dev", POLICIES)


def test_case_ids_may_not_repeat(tmp_path):
    extra = {"case_id": "dev_001", "ordered_service": "o", "service_code": "27447", "expected": "no_match"}
    with pytest.raises(ValueError, match="repeat"):
        routing_cases(copy_gold(tmp_path, extra), "dev", POLICIES)


def test_every_route_right_with_the_expected_distances(dev):
    results = run(dev, CaseIndex(dev), POLICIES)
    assert all(r.correct for r in results), report(results)
    assert report(results).endswith(
        f"Routed correctly: {len(dev)} of {len(dev)} (100%)\nUnsafe routes (reached a policy they should not): 0"
    )


def test_an_escalation_that_could_have_matched_is_a_safe_miss(dev):
    lands = LANDS | {"dev_002": (FACET, 0.9)}  # clinic shorthand lands too far away
    [miss] = [r for r in run(dev, CaseIndex(dev, lands), POLICIES) if not r.correct]
    assert (miss.case.case_id, miss.route.outcome, miss.unsafe) == ("dev_002", Outcome.NO_MATCH, False)


def test_a_wrong_escalation_reason_is_marked_with_the_closest_lcd(dev):
    lands = LANDS | {"route_dev_004": (FACET, 0.5)}  # the knee replacement lands near enough to the facet LCD
    text = report(run(dev, CaseIndex(dev, lands), POLICIES))
    [line] = [line for line in text.splitlines() if line.startswith("WRONG")]
    assert (
        line.split() == "WRONG route_dev_004 want no_match got code_not_covered closest LCD L38773 v32 at 0.500".split()
    )
    n = len(dev)
    assert f"Routed correctly: {n - 1} of {n} ({(n - 1) / n:.0%})" in text


def test_reaching_a_policy_it_should_not_is_unsafe():
    case = RoutingCase(case_id="c", ordered_service="o", service_code="64493", expected="no_match")
    reached = Route(outcome=Outcome.MATCHED, policy_id="l38773_first_mbb", reason="r")
    result = RoutingResult(case=case, route=reached)
    assert result.unsafe and not result.correct
    assert report([result]).splitlines()[0].startswith("UNSAFE c ")


def test_the_gate_takes_a_safe_miss_but_not_below_the_threshold(dev, capsys):
    one_miss = lambda sources: CaseIndex(dev, LANDS | {"dev_002": (FACET, 0.9)})  # noqa: E731
    assert main(["--min-accuracy", "0.95"], make_index=one_miss) == 0
    assert main(["--min-accuracy", "1.0"], make_index=one_miss) == 1
    assert main(["--min-accuracy", "0.95"], make_index=lambda sources: FakeIndex((VEINS, 0.9))) == 1
    assert f"Routed correctly: 2 of {len(dev)} " in capsys.readouterr().out  # only the two no-match cases


def test_any_unsafe_route_fails_the_gate_whatever_the_accuracy(tmp_path):
    extra = {"case_id": "x", "ordered_service": "Knee replacement", "service_code": "64493", "expected": "no_match"}
    gold = copy_gold(tmp_path, extra)
    cases = routing_cases(gold, "dev", POLICIES)
    assert main(["--gold", str(gold)], make_index=lambda sources: CaseIndex(cases)) == 1  # x lands on facet


def test_the_index_is_built_from_every_lcd_text(dev):
    seen = {}
    main([], make_index=lambda sources: seen.setdefault("sources", sources) and CaseIndex(dev))
    assert set(seen["sources"]) == {FACET, VEINS, EYELID}


def test_reaching_another_policy_is_wrong():
    case = RoutingCase(case_id="c", ordered_service="o", service_code="64493", expected="matched", policy_id="a")
    reached = Route(outcome=Outcome.MATCHED, policy_id="b", reason="r")
    assert not RoutingResult(case=case, route=reached).correct


class WordsIndex(CaseIndex):
    """CaseIndex, with a keyword score for each order's LCD: zero unless words gives one by case ID."""

    def __init__(self, cases, lands=LANDS, words=None):
        super().__init__(cases, lands)
        self.words = {c.ordered_service: (words or {}).get(c.case_id, 0.0) for c in cases}

    def search(self, text):
        return [c.model_copy(update={"keyword": self.words[text]}) for c in super().search(text)]


def test_the_keyword_minimum_sits_between_the_dev_orders_it_was_chosen_on(dev):
    keywords = KeywordIndex(load_sources(Path("policies")))
    best = {c.case_id: round(max(keywords.scores(c.ordered_service).values()), 2) for c in dev}
    assert (best["route_dev_005"], best["dev_002"]) == (0.27, 0.43)  # cataract surgery; the shorthand miss
    must_not_match = [best[c.case_id] for c in dev if c.expected is Outcome.NO_MATCH]
    assert max(must_not_match) == 0.27 and MIN_KEYWORD == round((0.27 + 0.43) / 2, 2)


FAR = LANDS | {"dev_002": (FACET, 0.68), "route_dev_004": (FACET, 0.9)}
PROBE = {"case_id": "probe", "ordered_service": "Epidural injection", "service_code": "64493", "expected": "no_match"}


def test_compare_lists_what_hybrid_fixes_breaks_and_makes_unsafe(dev):
    probe = RoutingCase.model_validate(PROBE)
    cases = [*dev, probe]
    index = WordsIndex(cases, FAR | {"probe": (FACET, 0.7)}, {"dev_002": 0.43, "route_dev_004": 0.5, "probe": 0.6})
    embedding, hybrid = (run(cases, index, POLICIES, r) for r in Retrieval)
    text = compare_report(embedding, hybrid)
    n = len(cases)
    # embedding misses dev_002; hybrid fixes it but breaks route_dev_004 and reaches the probe
    assert text.splitlines()[1].split() == ["Routed", "correctly", str(n - 1), "of", str(n), str(n - 2), "of", str(n)]
    assert text.splitlines()[2].split() == ["Unsafe", "routes", "0", "1"]
    assert "fixed  dev_002" in text and "broke  route_dev_004" in text and "UNSAFE probe" in text
    assert "  dev_002            LCD L38773 v32 0.43  Dx MBB L4-5, L5-S1" in text.splitlines()
    assert "(reached by keyword, 0.43)" in report(hybrid) and "reached by keyword" not in report(embedding)


def test_the_comparison_fails_on_an_unsafe_route_in_either_mode(tmp_path, capsys):
    gold = copy_gold(tmp_path, PROBE)
    cases = routing_cases(gold, "dev", POLICIES)
    make = lambda words: lambda sources: WordsIndex(cases, FAR | {"probe": (FACET, 0.7)}, words)  # noqa: E731
    assert main(["--gold", str(gold), "--compare"], make_index=make({"probe": 0.6})) == 1
    assert main(["--gold", str(gold), "--compare"], make_index=make({"probe": 0.2})) == 0
    assert main(["--gold", str(gold), "--retrieval", "hybrid"], make_index=make({"probe": 0.6})) == 1
    assert main(["--gold", str(gold)], make_index=make({"probe": 0.6})) == 0  # embedding-only never reaches it
