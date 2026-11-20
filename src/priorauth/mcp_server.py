"""PriorAuth-AI as an MCP server: policy lookup, synthetic charts and the decision, as tools for any MCP client.

The model judges and code decides, here as in the pipeline. The client's model (Claude in Claude Code or
Claude Desktop) reads a chart and judges each criterion with verbatim quotes; check_findings verifies every
quote against its note and makes the decision with the same code the agent's runs go through. The server
never calls a model, stores nothing and needs no API key.

    python -m priorauth.mcp_server          # stdio; needs the mcp and retrieval extras

Tools: list_policies, get_policy, find_policy, list_cases, get_case, check_findings. The review_case prompt
runs the whole workflow on one case. Charts are the synthetic dev cases, without their gold labels; held-out
cases are not served. A chart passed in by the caller is checked in memory and never saved.
"""

import os
from collections.abc import Callable
from functools import cache
from pathlib import Path
from typing import Annotated

import yaml
from mcp.server import MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from mcp.types import ToolAnnotations
from pydantic import Field

from priorauth.pipeline import determine
from priorauth.policies import fingerprint, load_policies, load_sources
from priorauth.retrieval import PolicyIndex, Route, lcd_name, route
from priorauth.schemas import Contract, CriterionResult, Decision, PARequest, Policy

# The repository root: policies/ and evals/gold/dev/ are read from here. Set PRIORAUTH_ROOT when the package
# is not installed in editable mode from a clone.
ROOT = Path(os.environ.get("PRIORAUTH_ROOT") or Path(__file__).resolve().parents[2])
READ_ONLY = ToolAnnotations(read_only_hint=True, open_world_hint=False)

INSTRUCTIONS = """\
PriorAuth-AI checks whether a prior authorization request is ready to submit under a Medicare coverage policy.
You judge, code decides: read the chart, judge every criterion of the policy as met, not_met or
insufficient_evidence, and quote the chart verbatim for each judgment. Then call check_findings, which checks
every quote against its note and returns the decision. Report that decision; never make one yourself.
Use synthetic charts only: never send real patient information to these tools."""


class PolicySummary(Contract):
    policy_id: str
    title: str
    coverage_document: str  # "LCD L38773 v32"
    service_codes: list[str]
    fingerprint: str  # changes whenever a criterion, quote or code changes
    criteria: list[str]


class CaseSummary(Contract):
    case_id: str
    service_code: str
    ordered_service: str | None
    request_date: str
    notes: int


class Check(Contract):
    """What code decided from the findings, and why."""

    decision: Decision
    rationale: str
    missing_documentation: list[str]  # what the clinic should add, one line per insufficient criterion
    rejected_quotes: list[str]  # quotes not found in the note they name; each one's finding was dropped
    unchecked: list[str]  # criteria with no finding: a required one escalates the case
    verified: list[str]  # criteria whose findings passed the quote check
    policy_fingerprint: str


def dev_charts(root: Path) -> dict[str, PARequest]:
    """The synthetic dev requests. Only the request is read: the gold labels never leave the file."""
    charts = {}
    for path in sorted((root / "evals" / "gold" / "dev").glob("*.yaml")):
        request = PARequest.model_validate(yaml.safe_load(path.read_text(encoding="utf-8"))["request"])
        charts[request.case_id] = request
    return charts


def create_server(root: Path = ROOT, index_factory: Callable[[dict[str, str]], PolicyIndex] = PolicyIndex) -> MCPServer:
    policies = load_policies(root / "policies")  # every criterion's LCD quotes are checked here, at startup
    charts = dev_charts(root)
    index = cache(lambda: index_factory(load_sources(root / "policies")))  # built on the first find_policy
    server = MCPServer("priorauth", instructions=INSTRUCTIONS)

    def policy_named(policy_id: str) -> Policy:
        if policy_id not in policies:
            raise ToolError(f"unknown policy {policy_id!r}; list_policies gives the IDs: {sorted(policies)}")
        return policies[policy_id]

    def chart_named(case_id: str) -> PARequest:
        if case_id not in charts:
            raise ToolError(f"unknown case {case_id!r}; only the synthetic dev cases are served (list_cases)")
        return charts[case_id]

    @server.tool(annotations=READ_ONLY)
    def list_policies() -> list[PolicySummary]:
        """The coverage policies this server decides, each built on one Medicare LCD version."""
        return [
            PolicySummary(
                policy_id=p.policy_id,
                title=p.title,
                coverage_document=lcd_name(p.source),
                service_codes=p.service_codes,
                fingerprint=fingerprint(p),
                criteria=[c.criterion_id for c in p.criteria],
            )
            for p in policies.values()
        ]

    @server.tool(annotations=READ_ONLY)
    def get_policy(policy_id: str) -> Policy:
        """One policy's criteria: the text to judge each against, whether it is required, and the LCD's words."""
        return policy_named(policy_id)

    @server.tool(annotations=READ_ONLY)
    def find_policy(
        ordered_service: Annotated[str, Field(description="The service in the ordering clinician's words")],
        service_code: Annotated[str, Field(description="The CPT code requested, e.g. 64493")],
    ) -> Route:
        """Find the policy for an order: retrieval over the LCD texts proposes, then code checks the service code.

        Returns the matched policy, or the reason the request must go to a person. The first call builds the
        index and may download the embedding model.
        """
        return route(index(), policies.values(), ordered_service, service_code)

    @server.tool(annotations=READ_ONLY)
    def list_cases() -> list[CaseSummary]:
        """The synthetic dev cases you can review. Their gold labels are never served."""
        return [
            CaseSummary(
                case_id=r.case_id,
                service_code=r.service_code,
                ordered_service=r.ordered_service,
                request_date=r.request_date.isoformat(),
                notes=len(r.notes),
            )
            for r in charts.values()
        ]

    @server.tool(annotations=READ_ONLY)
    def get_case(case_id: str) -> PARequest:
        """One synthetic request: the order, the codes and every note in the chart."""
        return chart_named(case_id)

    @server.tool(annotations=READ_ONLY)
    def check_findings(
        policy_id: str,
        findings: Annotated[
            list[CriterionResult],
            Field(description="One judgment per criterion, each met finding quoting the chart verbatim"),
        ],
        case_id: Annotated[str | None, Field(description="A dev case to check against (list_cases)")] = None,
        request: Annotated[PARequest | None, Field(description="Or your own synthetic request")] = None,
    ) -> Check:
        """Check findings against the chart and decide, in code: the same verify and decide steps as the agent.

        A finding with any quote not in the note it names is dropped. The decision is ready_to_submit only when
        every required criterion is met with verified quotes; needs_documentation when evidence is missing but
        nothing contradicts the policy; escalate otherwise.
        """
        if (case_id is None) == (request is None):
            raise ToolError("give exactly one of case_id or request")
        policy = policy_named(policy_id)
        chart = chart_named(case_id) if case_id is not None else request
        if chart.service_code not in policy.service_codes:
            raise ToolError(f"{policy_id} does not cover code {chart.service_code}; find_policy finds the policy")
        known = {c.criterion_id for c in policy.criteria}
        ids = [f.criterion_id for f in findings]
        if unknown := sorted(set(ids) - known):
            raise ToolError(f"criteria not in {policy_id}: {unknown}; get_policy lists its criteria")
        if repeated := sorted({cid for cid in ids if ids.count(cid) > 1}):
            raise ToolError(f"one finding per criterion; repeated: {repeated}")
        unchecked = [c.criterion_id for c in policy.criteria if c.criterion_id not in ids]
        determination, problems = determine(policy, chart, findings, unchecked)
        return Check(
            decision=determination.decision,
            rationale=determination.rationale,
            missing_documentation=determination.missing_documentation,
            rejected_quotes=problems,
            unchecked=unchecked,
            verified=[r.criterion_id for r in determination.criteria],
            policy_fingerprint=fingerprint(policy),
        )

    @server.prompt()
    def review_case(case_id: str) -> str:
        """Review one synthetic case end to end: find the policy, judge each criterion, let code decide."""
        return (
            f"Review prior authorization case {case_id} with the priorauth tools.\n"
            f"1. get_case {case_id}, then find_policy with its ordered_service and service_code. If no policy "
            "matches, report the reason and stop.\n"
            "2. get_policy, then judge every criterion from the chart alone: met, not_met or "
            "insufficient_evidence. Quote each note verbatim, at least two words, with its note_id.\n"
            f"3. check_findings with the policy_id, case_id {case_id} and one finding per criterion. If quotes "
            "are rejected, fix them and check again.\n"
            "4. Report the decision check_findings returned, its rationale and any missing documentation, with "
            "a short table of your findings. Do not change or second-guess the decision."
        )

    return server


def main() -> None:
    create_server().run()  # stdio: the client starts this process and talks over stdin and stdout


if __name__ == "__main__":
    main()
