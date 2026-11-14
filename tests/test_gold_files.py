"""Every committed gold case must load against the real policies, in both splits."""

import re
from pathlib import Path

import pytest

from evals.gold import SPLITS, load_split
from priorauth.policies import load_policies

ROOT = Path(__file__).resolve().parents[1]
GOLD = ROOT / "evals" / "gold"
POLICIES = load_policies(ROOT / "policies")


@pytest.mark.parametrize("split", SPLITS)
def test_cases_load_and_their_labels_are_consistent(split):
    assert load_split(GOLD, split, POLICIES)


@pytest.mark.parametrize("split", SPLITS)
def test_case_ids_do_not_hint_at_the_answer(split):
    """The case ID travels with the request to the model, so it must be a neutral number."""
    for case_id in load_split(GOLD, split, POLICIES):
        assert re.fullmatch(rf"{split}_\d{{3}}", case_id), case_id


def test_held_out_charts_do_not_reuse_dev_notes():
    """A held-out note copied from a dev case would leak material the agent was tuned on."""
    dev_notes = {note.text for case in load_split(GOLD, "dev", POLICIES).values() for note in case.request.notes}
    for case in load_split(GOLD, "heldout", POLICIES).values():
        assert not dev_notes & {note.text for note in case.request.notes}, case.case_id
