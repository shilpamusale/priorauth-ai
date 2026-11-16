"""The HTTP API, with a scripted model: no API key, no network."""

import datetime
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from test_retrieval import FakeIndex

from evals.gold import load_split
from priorauth.api import DailyLimit, create_app, seconds_until_tomorrow
from priorauth.llm.base import ModelResponse, StopReason, ToolCall
from priorauth.llm.scripted import ScriptedClient
from priorauth.pipeline import AuditRecord
from priorauth.policies import fingerprint, load_policies
from priorauth.retrieval import Outcome

POLICIES = load_policies(Path("policies"))
CASE = load_split(Path("evals/gold"), "dev", POLICIES)["dev_001"]
POLICY = POLICIES[CASE.policy_id]


def gaps_reply(policy=POLICY):
    """A model reply that submits every criterion as insufficient evidence: a valid run with no quotes."""
    findings = [
        {"criterion_id": c.criterion_id, "status": "insufficient_evidence", "evidence": [], "reasoning": "Not found."}
        for c in policy.criteria
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
    listed = {p["policy_id"]: p for p in service()[0].get("/policies").json()}["l38773_first_mbb"]
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
    assert set(paths) == {"/health", "/policies", "/requests", "/determinations"}


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


# Routed requests: retrieval picks the policy, so these tests give the service a fake index.
FACET_LCD = POLICY.source
VEIN_LCD = "https://www.cms.gov/medicare-coverage-database/view/lcd.aspx?lcdid=34082&ver=31"


def routed_service(*candidates, runs=5, daily_runs=50, policy=POLICY):
    model = ScriptedClient([gaps_reply(policy) for _ in range(runs)])
    built = []

    def index_factory(sources):
        built.append(sorted(sources))
        return FakeIndex(*candidates)

    app = create_app(Path("policies"), daily_runs=daily_runs, client_factory=lambda: model, index_factory=index_factory)
    return TestClient(app), model, built


def routed_body(**changes):
    return CASE.request.model_dump(mode="json") | changes


def test_a_routed_request_reaches_its_policy_and_runs():
    client, model, _ = routed_service((FACET_LCD, 0.3))
    response = client.post("/requests", json=routed_body())
    assert response.status_code == 200
    routed = response.json()
    assert routed["route"]["outcome"] == "matched" and routed["route"]["policy_id"] == "l38773_first_mbb"
    assert routed["audit"]["determination"]["decision"] == "needs_documentation" and model.requests


@pytest.mark.parametrize(
    "candidates,changes,outcome",
    [
        ([(FACET_LCD, 0.95)], {}, Outcome.NO_MATCH),
        ([(VEIN_LCD, 0.2)], {}, Outcome.UNSUPPORTED),
        ([(FACET_LCD, 0.2)], {"service_code": "27447"}, Outcome.CODE_NOT_COVERED),
        ([(FACET_LCD, 0.2)], {"ordered_service": None}, Outcome.NO_ORDER),
    ],
)
def test_a_request_that_reaches_no_policy_escalates_without_a_model_call_or_a_run(candidates, changes, outcome):
    client, model, _ = routed_service(*candidates, daily_runs=0)  # no runs left: an escalation must not need one
    routed = client.post("/requests", json=routed_body(**changes)).json()
    assert (routed["route"]["outcome"], routed["route"]["policy_id"], routed["audit"]) == (outcome.value, None, None)
    assert routed["route"]["reason"] and model.requests == []


def test_a_routed_run_counts_against_the_daily_limit():
    client, _, _ = routed_service((FACET_LCD, 0.3), daily_runs=1)
    assert client.post("/requests", json=routed_body()).status_code == 200
    second = client.post("/requests", json=routed_body())
    assert second.status_code == 429 and int(second.headers["Retry-After"]) > 0


def test_the_index_is_built_once_on_the_first_routed_request_from_every_lcd_text():
    client, _, built = routed_service((FACET_LCD, 0.95))
    client.get("/health")
    assert built == []  # starting the service does not build it
    client.post("/requests", json=routed_body())
    client.post("/requests", json=routed_body())
    assert len(built) == 1 and FACET_LCD in built[0] and VEIN_LCD in built[0]


def test_a_named_policy_that_does_not_cover_the_code_is_422_and_uses_no_run():
    client, model = service(daily_runs=1)
    request = CASE.request.model_dump(mode="json") | {"service_code": "27447"}
    response = client.post("/determinations", json={"policy_id": "l38773_first_mbb", "request": request})
    assert response.status_code == 422 and "does not cover code 27447" in response.json()["detail"]
    assert model.requests == [] and client.post("/determinations", json=body()).status_code == 200


def test_the_run_uses_the_policy_the_route_reached():
    eyelid = POLICIES["l33944_upper_bleph"]
    client, _, _ = routed_service((eyelid.source, 0.2), policy=eyelid)
    routed = client.post("/requests", json=routed_body(service_code="15823")).json()
    assert routed["route"]["policy_id"] == "l33944_upper_bleph"
    assert routed["audit"]["determination"]["policy_id"] == "l33944_upper_bleph"
