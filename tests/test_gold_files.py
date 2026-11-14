"""Every committed gold case must load against the real policies."""

import re
from pathlib import Path

from evals.gold import load_split
from priorauth.policies import load_policies

ROOT = Path(__file__).resolve().parents[1]
GOLD = ROOT / "evals" / "gold"
POLICIES = load_policies(ROOT / "policies")


def test_dev_cases_load_and_their_labels_are_consistent():
    assert load_split(GOLD, "dev", POLICIES)


def test_case_ids_do_not_hint_at_the_answer():
    """The case ID travels with the request to the model, so it must be a neutral number."""
    for case_id in load_split(GOLD, "dev", POLICIES):
        assert re.fullmatch(r"dev_\d{3}", case_id), case_id
