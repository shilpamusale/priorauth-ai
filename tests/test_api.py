"""The HTTP API, with a scripted model: no API key, no network."""

import datetime
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from evals.gold import load_split
from priorauth.api import DailyLimit, create_app, seconds_until_tomorrow
from priorauth.llm.base import ModelResponse, StopReason, ToolCall
from priorauth.llm.scripted import ScriptedClient
from priorauth.pipeline import AuditRecord
from priorauth.policies import fingerprint, load_policies

POLICIES = load_policies(Path("policies"))
CASE = load_split(Path("evals/gold"), "dev", POLICIES)["dev_001"]
POLICY = POLICIES[CASE.policy_id]


def gaps_reply():
    """A model reply that submits every criterion as insufficient evidence: a valid run with no quotes."""
    findings = [
        {"criterion_id": c.criterion_id, "status": "insufficient_evidence", "evidence": [], "reasoning": "Not found."}
        for c in POLICY.criteria
    ]
    calls = [ToolCall(id=f"toolu_{i}", name="submit_finding", input=f) for i, f in enumerate(findings)]
    return ModelResponse(tool_calls=calls, stop_reason=StopReason.TOOL_USE)


def service(runs=5, daily_runs=50, today=lambda: datetime.date(2026, 11, 14)):
    model = ScriptedClient([gaps_reply() for _ in range(runs)])
    app = create_app(Path("policies"), daily_runs=daily_runs, client_factory=lambda: model, today=today)
    return TestClient(app), model


def body(policy_id="l38773_first_mbb"):
    return {"policy_id": policy_id, "request": CASE.request.model_dump(mode="json")}


def test_health():
    assert service()[0].get("/health").json() == {"status": "ok"}


def test_policies_are_listed_with_their_fingerprints():
    [listed] = service()[0].get("/policies").json()
    assert listed["policy_id"] == "l38773_first_mbb"
    assert listed["fingerprint"] == fingerprint(POLICY)
    assert listed["criteria"] == [c.criterion_id for c in POLICY.criteria]


def test_a_request_returns_its_determination_and_audit_record():
    client, model = service()
    response = client.post("/determinations", json=body())
    assert response.status_code == 200
    record = AuditRecord.model_validate(response.json())
    assert record.determination.case_id == "dev_001"
    assert record.determination.decision == "needs_documentation"
    assert len(record.determination.missing_documentation) == len(POLICY.criteria)
    assert record.policy_fingerprint == fingerprint(POLICY)
    assert model.requests[0].model == "claude-sonnet-5-5"


def test_unknown_policy_is_404_and_uses_no_run():
    client, model = service(daily_runs=1)
    assert client.post("/determinations", json=body("no_such_policy")).status_code == 404
    assert model.requests == []
    assert client.post("/determinations", json=body()).status_code == 200  # the daily run was not used


def test_a_malformed_request_is_422_and_uses_no_run():
    client, model = service(daily_runs=1)
    bad = body()
    bad["request"]["icd10_codes"] = ["not-a-code"]
    assert client.post("/determinations", json=bad).status_code == 422
    assert model.requests == []
    assert client.post("/determinations", json=body()).status_code == 200


def test_the_daily_limit_returns_429_with_retry_after():
    client, _ = service(daily_runs=2)
    assert [client.post("/determinations", json=body()).status_code for _ in range(3)] == [200, 200, 429]
    response = client.post("/determinations", json=body())
    assert response.json()["detail"] == "daily run limit reached"
    assert 0 < int(response.headers["Retry-After"]) <= 24 * 3600


def test_the_limit_resets_each_utc_day():
    day = [datetime.date(2026, 11, 14)]
    limit = DailyLimit(1, today=lambda: day[0])
    assert limit.take() and not limit.take()
    day[0] = datetime.date(2026, 11, 15)
    assert limit.take() and not limit.take()


def test_seconds_until_tomorrow_is_within_a_day():
    assert 1 <= seconds_until_tomorrow() <= 24 * 3600


def test_the_model_client_is_created_once():
    made = []

    def factory():
        made.append(1)
        return ScriptedClient([gaps_reply(), gaps_reply()])

    client = TestClient(create_app(Path("policies"), client_factory=factory))
    client.post("/determinations", json=body())
    client.post("/determinations", json=body())
    assert made == [1]


def test_openapi_docs_describe_the_endpoints():
    paths = service()[0].get("/openapi.json").json()["paths"]
    assert set(paths) == {"/health", "/policies", "/determinations"}


@pytest.mark.parametrize("missing", ["policy_id", "request"])
def test_both_fields_are_required(missing):
    payload = body()
    del payload[missing]
    assert service()[0].post("/determinations", json=payload).status_code == 422


def test_the_deployment_sets_the_daily_limit_by_environment(monkeypatch):
    monkeypatch.setenv("PRIORAUTH_DAILY_RUNS", "1")
    model = ScriptedClient([gaps_reply()])
    client = TestClient(create_app(client_factory=lambda: model))
    assert [client.post("/determinations", json=body()).status_code for _ in range(2)] == [200, 429]
