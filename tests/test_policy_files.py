"""Every policy in policies/ must pass the loader's rules, including quotes checked against its LCD text."""

from pathlib import Path

from priorauth.policies import load_policies

POLICY_DIR = Path(__file__).resolve().parents[1] / "policies"


def test_every_policy_file_loads_and_its_quotes_match_the_source():
    assert load_policies(POLICY_DIR)

def test_every_policy_cites_a_public_source():
    for policy in load_policies(POLICY_DIR).values():
        assert policy.source.startswith("https://"), f"{policy.policy_id} must cite its coverage document"


def test_facet_policy_keeps_the_eight_reviewed_criteria():
    """Changing the reviewed criteria should be a deliberate edit to this test, not an accident."""
    policy = load_policies(POLICY_DIR)["l38773_first_mbb"]
    assert len(policy.criteria) == 8 and all(c.required for c in policy.criteria)