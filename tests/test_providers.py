"""The provider layer: the same Claude client on the Claude API or on Amazon Bedrock."""

import shutil
import sys
import types
from pathlib import Path

import pytest
from test_claude import REQUEST, FakeSDK, message

import priorauth.llm.providers as providers
from evals.providers import compare, main, report
from evals.runner import GOLD, RECORDINGS, Mode, provider_root, run_split
from priorauth.llm.claude import ClaudeClient
from priorauth.llm.providers import BEDROCK_MODEL_IDS, Provider, bedrock_token, bedrock_url, make_client
from priorauth.policies import load_policies
from priorauth.schemas import CriterionStatus, Decision

REPLY = message([{"type": "text", "text": "Done."}])


@pytest.fixture
def no_aws(monkeypatch):
    for name in ("AWS_BEARER_TOKEN_BEDROCK", "AWS_REGION"):
        monkeypatch.delenv(name, raising=False)


def test_bedrock_is_reached_at_its_anthropic_endpoint():
    assert bedrock_url("us-west-2") == "https://bedrock-runtime.us-west-2.amazonaws.com/anthropic"


def test_a_bedrock_api_key_comes_from_the_environment_first(monkeypatch, no_aws):
    monkeypatch.setenv("AWS_BEARER_TOKEN_BEDROCK", "  bedrock-key  ")
    monkeypatch.setitem(sys.modules, "aws_bedrock_token_generator", None)  # never needed when a key is set
    assert bedrock_token("us-east-1") == "bedrock-key"


def test_without_a_key_a_short_lived_one_is_made_from_aws_credentials(monkeypatch, no_aws):
    made = []
    generator = types.SimpleNamespace(provide_token=lambda region: made.append(region) or "short-lived")
    monkeypatch.setitem(sys.modules, "aws_bedrock_token_generator", generator)
    assert bedrock_token("eu-west-1") == "short-lived" and made == ["eu-west-1"]


def test_with_neither_the_error_says_what_to_set(monkeypatch, no_aws):
    monkeypatch.setitem(sys.modules, "aws_bedrock_token_generator", None)
    with pytest.raises(RuntimeError, match="AWS_BEARER_TOKEN_BEDROCK"):
        bedrock_token("us-east-1")


def test_the_claude_api_client_is_the_default(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    client = make_client(Provider.ANTHROPIC)
    assert isinstance(client, ClaudeClient) and client._model_ids is None
    assert "bedrock" not in str(client._sdk.base_url)


def test_the_bedrock_client_points_at_bedrock_with_its_key_and_model_ids(monkeypatch, no_aws):
    monkeypatch.setenv("AWS_BEARER_TOKEN_BEDROCK", "bedrock-key")
    client = make_client(Provider.BEDROCK)
    assert str(client._sdk.base_url).rstrip("/") == bedrock_url("us-east-1")
    assert client._sdk.api_key == "bedrock-key" and client._model_ids == BEDROCK_MODEL_IDS
    monkeypatch.setenv("AWS_REGION", "us-west-2")
    assert "us-west-2" in str(make_client(Provider.BEDROCK)._sdk.base_url)
    assert "eu-west-1" in str(make_client(Provider.BEDROCK, region="eu-west-1")._sdk.base_url)


def test_the_request_keeps_the_model_name_and_bedrock_gets_its_own_id():
    sdk = FakeSDK(REPLY)
    ClaudeClient(sdk, model_ids=BEDROCK_MODEL_IDS).complete(REQUEST)
    assert REQUEST.model == "claude-sonnet-5-5" and sdk.sent[0]["model"] == "global.anthropic.claude-sonnet-5-5"
    plain = FakeSDK(REPLY)
    ClaudeClient(plain).complete(REQUEST)
    assert plain.sent[0]["model"] == "claude-sonnet-5-5"


def test_a_model_with_no_bedrock_id_fails_before_any_call():
    sdk = FakeSDK(REPLY)
    with pytest.raises(ValueError, match="no provider model ID for 'claude-sonnet-5-5'"):
        ClaudeClient(sdk, model_ids={"claude-opus-5-5": "x"}).complete(REQUEST)
    assert sdk.sent == []


def test_the_api_chooses_its_provider_from_the_environment(monkeypatch):
    from priorauth.api import claude_client

    chosen = []
    monkeypatch.setattr(providers, "make_client", lambda provider: chosen.append(provider) or "client")
    monkeypatch.delenv("PRIORAUTH_PROVIDER", raising=False)
    claude_client()
    monkeypatch.setenv("PRIORAUTH_PROVIDER", "bedrock")
    claude_client()
    assert chosen == [Provider.ANTHROPIC, Provider.BEDROCK]


def test_each_provider_records_and_replays_in_its_own_folder(tmp_path):
    assert provider_root(tmp_path, Provider.ANTHROPIC) == tmp_path
    assert provider_root(tmp_path, Provider.BEDROCK) == tmp_path / "bedrock"


POLICIES = load_policies(Path("policies"))


@pytest.fixture(scope="module")
def api_runs():
    return run_split("dev", Mode.REPLAY, POLICIES, gold=GOLD, recordings=RECORDINGS)


def test_identical_runs_match_on_every_decision_and_criterion(api_runs):
    from evals.gold import load_split

    comparison = compare({Provider.ANTHROPIC: api_runs, Provider.BEDROCK: api_runs}, load_split(GOLD, "dev", POLICIES))
    assert comparison.decision_differences == [] and comparison.criterion_differences == []
    assert comparison.compared_criteria == 372
    text = report(comparison)
    assert "Decisions that differ between providers: 0" in text and "Criterion statuses that differ: 0 of 372" in text


def test_differences_are_listed_by_case_trial_and_criterion(api_runs):
    from evals.gold import load_split

    first = api_runs[0]
    changed = first.record.determination.model_copy(
        update={
            "decision": Decision.ESCALATE,
            "criteria": [
                r.model_copy(update={"status": CriterionStatus.NOT_MET, "evidence": []}) if i == 0 else r
                for i, r in enumerate(first.record.determination.criteria)
            ],
        }
    )
    other = [
        first.model_copy(update={"record": first.record.model_copy(update={"determination": changed})}),
        *api_runs[1:],
    ]
    comparison = compare({Provider.ANTHROPIC: api_runs, Provider.BEDROCK: other}, load_split(GOLD, "dev", POLICIES))
    assert comparison.decision_differences == [f"{first.case_id} trial 1: anthropic ready_to_submit, bedrock escalate"]
    [criterion] = comparison.criterion_differences
    assert criterion.startswith(
        f"{first.case_id} trial 1 l38773_first_mbb.pain_severity: anthropic met, bedrock not_met"
    )


def test_the_command_line_compares_and_fails_when_a_provider_has_no_recordings(tmp_path, capsys):
    recordings = tmp_path / "rec"
    shutil.copytree(RECORDINGS, recordings, ignore=shutil.ignore_patterns("judge"))
    assert main(["--recordings", str(recordings)]) == 1  # no Bedrock recordings yet
    missing = capsys.readouterr().out
    assert "bedrock: 51 of 51 runs have no usable recording" in missing
    assert ["decision", "accuracy", "1.00", "-"] in [line.split() for line in missing.splitlines()]
    shutil.copytree(RECORDINGS / "dev", recordings / "bedrock" / "dev")
    assert main(["--recordings", str(recordings)]) == 0
    out = capsys.readouterr().out
    assert "Decisions that differ between providers: 0" in out
