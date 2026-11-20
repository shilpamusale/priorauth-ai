"""The MCP server: the tools a client's model calls, and the decision code makes from its findings."""

import json
import sys
from pathlib import Path

import anyio
import pytest
from embedding import WordHashEmbedding
from mcp import Client, StdioServerParameters
from mcp.client.stdio import get_default_environment

from evals.runner import GOLD, RECORDINGS, Mode, run_split
from priorauth.mcp_server import create_server
from priorauth.policies import fingerprint, load_policies, load_sources
from priorauth.retrieval import PolicyIndex, route
from priorauth.schemas import PARequest

ROOT = Path(__file__).resolve().parents[1]
POLICIES = load_policies(ROOT / "policies")
FACET = "l38773_first_mbb"
ORDER = "Diagnostic lumbar medial branch blocks L4-L5"


def index(sources):
    return PolicyIndex(sources, WordHashEmbedding())


@pytest.fixture(scope="module")
def server():
    return create_server(ROOT, index_factory=index)


def calls(server, *requests):
    """Results of tool calls made in one session, in order."""

    async def go():
        async with Client(server) as client:
            return [await client.call_tool(name, arguments) for name, arguments in requests]

    return anyio.run(go)


def call(server, name, **arguments):
    [result] = calls(server, (name, arguments))
    return result


def ok(result):
    assert not result.is_error, result.content[0].text
    content = result.structured_content
    return content["result"] if set(content) == {"result"} else content


def error(result):
    assert result.is_error
    return result.content[0].text


@pytest.fixture(scope="module")
def recorded():
    """The recorded single-agent dev runs, replayed: what the agent found and what code decided."""
    runs = run_split("dev", Mode.REPLAY, POLICIES, gold=GOLD, recordings=RECORDINGS)
    assert runs and not [r for r in runs if r.error]
    return runs


def findings_of(run):
    return [r.model_dump(mode="json") for r in run.record.determination.criteria]


def test_every_tool_is_read_only_and_says_what_it_does(server):
    async def go():
        async with Client(server) as client:
            return (await client.list_tools()).tools, (await client.list_prompts()).prompts

    tools, prompts = anyio.run(go)
    assert {t.name for t in tools} == {
        "list_policies",
        "get_policy",
        "find_policy",
        "list_cases",
        "get_case",
        "check_findings",
    }
    for tool in tools:
        assert tool.description and tool.annotations.read_only_hint and tool.annotations.open_world_hint is False
    assert [p.name for p in prompts] == ["review_case"]


def test_policies_are_listed_with_their_coverage_document_and_fingerprint(server):
    listed = {p["policy_id"]: p for p in ok(call(server, "list_policies"))}
    assert listed.keys() == POLICIES.keys()
    assert listed[FACET]["coverage_document"] == "LCD L38773 v32"
    assert listed["l33944_upper_bleph"]["coverage_document"] == "LCD L33944 v28"
    assert all(listed[pid]["fingerprint"] == fingerprint(policy) for pid, policy in POLICIES.items())


def test_a_policy_comes_with_its_criteria_and_an_unknown_one_is_refused(server):
    policy = ok(call(server, "get_policy", policy_id=FACET))
    assert [c["criterion_id"] for c in policy["criteria"]] == [c.criterion_id for c in POLICIES[FACET].criteria]
    assert all(c["source_quotes"] for c in policy["criteria"])
    assert "unknown policy 'knee'" in error(call(server, "get_policy", policy_id="knee"))


def test_the_cases_are_the_dev_charts_without_their_labels(server):
    cases = ok(call(server, "list_cases"))
    assert [c["case_id"] for c in cases] == sorted(p.stem for p in (GOLD / "dev").glob("*.yaml"))
    chart = ok(call(server, "get_case", case_id="dev_001"))
    assert set(chart) == set(PARequest.model_fields) and chart["notes"]
    served = json.dumps(chart)
    assert "labels" not in served and "expected_decision" not in served
    assert "only the synthetic dev cases" in error(call(server, "get_case", case_id="heldout_001"))


def test_find_policy_returns_the_route_retrieval_and_the_code_check_give(server):
    found = ok(call(server, "find_policy", ordered_service=ORDER, service_code="64493"))
    direct = route(index(load_sources(ROOT / "policies")), POLICIES.values(), ORDER, "64493")
    assert found == direct.model_dump(mode="json")
    assert found["outcome"] == "matched" and found["policy_id"] == FACET
    assert ok(call(server, "find_policy", ordered_service=ORDER, service_code="15822"))["outcome"] == "code_not_covered"


def test_code_decides_exactly_as_it_did_for_every_recorded_run(server, recorded):
    requests = [
        (
            "check_findings",
            {"policy_id": r.record.determination.policy_id, "case_id": r.case_id, "findings": findings_of(r)},
        )
        for r in recorded
    ]
    for run, result in zip(recorded, calls(server, *requests), strict=True):
        checked, recorded_determination = ok(result), run.record.determination
        assert checked["decision"] == recorded_determination.decision, (run.case_id, run.trial)
        assert checked["missing_documentation"] == recorded_determination.missing_documentation
        assert checked["policy_fingerprint"] == run.record.policy_fingerprint


def ready_run(recorded):
    return next(r for r in recorded if r.record.determination.decision == "ready_to_submit")


def test_a_quote_not_in_the_chart_is_rejected_and_the_case_escalates(server, recorded):
    run = ready_run(recorded)
    findings = findings_of(run)
    findings[0]["evidence"][0]["quote"] = "pain resolved completely after one visit"
    checked = ok(call(server, "check_findings", policy_id=FACET, case_id=run.case_id, findings=findings))
    assert checked["decision"] == "escalate"
    assert checked["rejected_quotes"] == [
        f"{findings[0]['criterion_id']}: quote 'pain resolved completely after one visit' is not in "
        f"{findings[0]['evidence'][0]['note_id']}"
    ]
    assert findings[0]["criterion_id"] not in checked["verified"]


def test_a_required_criterion_with_no_finding_escalates(server, recorded):
    run = ready_run(recorded)
    findings = findings_of(run)
    dropped = findings.pop()
    checked = ok(call(server, "check_findings", policy_id=FACET, case_id=run.case_id, findings=findings))
    assert checked["decision"] == "escalate" and checked["unchecked"] == [dropped["criterion_id"]]
    assert f"No finding: {dropped['criterion_id']}." in checked["rationale"]


def test_a_callers_own_synthetic_request_is_checked_the_same_way(server, recorded):
    run = ready_run(recorded)
    chart = ok(call(server, "get_case", case_id=run.case_id)) | {"case_id": "my_own_chart"}
    checked = ok(call(server, "check_findings", policy_id=FACET, request=chart, findings=findings_of(run)))
    assert checked["decision"] == "ready_to_submit"


def test_bad_input_is_reported_so_the_model_can_fix_it(server, recorded):
    run = ready_run(recorded)
    good = findings_of(run)
    extra = good + [dict(good[0])]
    unknown = good + [{"criterion_id": f"{FACET}.made_up", "status": "met", "evidence": good[0]["evidence"]}]
    no_quote = [dict(good[0], evidence=[])]
    chart = ok(call(server, "get_case", case_id=run.case_id)) | {"service_code": "15822"}
    common = {"policy_id": FACET, "findings": good}
    results = calls(
        server,
        ("check_findings", common),
        ("check_findings", common | {"case_id": run.case_id, "request": chart}),
        ("check_findings", common | {"case_id": run.case_id, "findings": unknown}),
        ("check_findings", common | {"case_id": run.case_id, "findings": extra}),
        ("check_findings", common | {"case_id": run.case_id, "findings": no_quote}),
        ("check_findings", common | {"request": chart}),
    )
    messages = [error(r) for r in results]
    assert "exactly one of case_id or request" in messages[0] and "exactly one" in messages[1]
    assert f"criteria not in {FACET}: ['{FACET}.made_up']" in messages[2]
    assert "one finding per criterion; repeated" in messages[3]
    assert "a criterion marked met must cite evidence" in messages[4]
    assert f"{FACET} does not cover code 15822" in messages[5]


def test_the_prompt_walks_the_client_through_one_case(server):
    async def go():
        async with Client(server) as client:
            return await client.get_prompt("review_case", {"case_id": "dev_004"})

    text = anyio.run(go).messages[0].content.text
    assert "get_case dev_004" in text and "check_findings" in text and "Do not change" in text


def test_the_server_starts_over_stdio_from_any_folder(tmp_path):
    """How Claude Code and Claude Desktop run it: a subprocess, started in a folder that is not the repo."""
    params = StdioServerParameters(
        command=sys.executable,
        args=["-m", "priorauth.mcp_server"],
        cwd=tmp_path,
        env=get_default_environment() | {"PYTHONPATH": str(ROOT / "src")},
    )

    async def go():
        async with Client(params) as client:
            return await client.call_tool("list_policies", {})

    assert {p["policy_id"] for p in ok(anyio.run(go))} == POLICIES.keys()
