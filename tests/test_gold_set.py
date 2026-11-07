from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from evals.schema import GoldSet, load_gold_set

GOLD = Path(__file__).resolve().parents[1] / "evals" / "gold" / "cases.yaml"


def test_gold_set_loads_and_is_self_consistent():
    gold = load_gold_set(GOLD)
    assert len(gold.cases) >= 4
    assert len({c.case_id for c in gold.cases}) == len(gold.cases)


def test_gold_set_covers_every_decision():
    assert {c.expected_decision.value for c in load_gold_set(GOLD).cases} == {
        "ready_to_submit",
        "needs_documentation",
        "escalate",
    }


def test_contradictory_gold_case_is_rejected():
    raw = yaml.safe_load(GOLD.read_text())
    raw["cases"][0]["expected_decision"] = "escalate"  # case 0 meets every criterion
    with pytest.raises(ValidationError, match="contradicts"):
        GoldSet.model_validate(raw)
