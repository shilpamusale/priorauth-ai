"""One case end to end, and the audit record that explains the result.

    match()   the model reads the chart with tools and submits findings
    verify()  findings whose quotes are not in the chart are dropped
    decide()  the decision, computed in code from the grounded findings only

The audit record keeps what is needed to explain or reproduce a decision: the determination, the policy
and prompt fingerprints, the model, every tool call and result, and what was skipped or rejected.
"""

import datetime
import hashlib
from enum import StrEnum

from pydantic import Field

from priorauth.decision import decide
from priorauth.llm.base import ModelClient, Turn, Usage
from priorauth.matcher import DEFAULT_MODEL, MAX_STEPS, MatchRun, StopCause, load_prompt, match
from priorauth.orchestrator import run_agents
from priorauth.policies import fingerprint
from priorauth.schemas import Contract, CriterionResult, CriterionStatus, Decision, Determination, PARequest, Policy
from priorauth.verify import verify


class Pipeline(StrEnum):
    SINGLE_AGENT = "single_agent"  # one matcher checks every criterion
    MULTI_AGENT = "multi_agent"  # one evidence agent per criterion, run in parallel by the orchestrator


class AuditRecord(Contract):
    determination: Determination
    policy_fingerprint: str
    model: str
    prompt_version: str
    prompt_fingerprint: str  # changes if the prompt file is edited without a new version
    stop: StopCause
    steps: int
    usage: Usage
    skipped: list[str]  # criteria with no finding when the matcher stopped
    rejected_quotes: list[str]  # quotes that failed verification, one line each
    turns: list[Turn]
    pipeline: Pipeline = Pipeline.SINGLE_AGENT
    recorded_at: datetime.datetime = Field(default_factory=lambda: datetime.datetime.now(datetime.UTC))


def missing_documentation(policy: Policy, results: list[CriterionResult]) -> list[str]:
    """What to add to the chart: one line per criterion it does not document enough, in the matcher's words."""
    texts = {c.criterion_id: c.text for c in policy.criteria}
    return [
        f"{r.criterion_id}: {r.reasoning or texts.get(r.criterion_id, '')}"
        for r in results
        if r.status is CriterionStatus.INSUFFICIENT_EVIDENCE
    ]


def rationale(
    decision: Decision, policy: Policy, results: list[CriterionResult], skipped: list[str], rejected: list[str]
) -> str:
    """Why the case got its decision, written from the facts decide() used."""
    if decision is Decision.READY_TO_SUBMIT:
        return "Ready to submit: every required criterion is met, with quotes verified against the chart."
    status = {r.criterion_id: r.status for r in results}
    required = [c.criterion_id for c in policy.criteria if c.required]
    reasons = {
        "Not met": [cid for cid in required if status.get(cid) is CriterionStatus.NOT_MET],
        "Evidence failed verification": [cid for cid in required if cid in rejected],
        "No finding": [cid for cid in required if cid in skipped],
        "Documentation missing": [cid for cid in required if status.get(cid) is CriterionStatus.INSUFFICIENT_EVIDENCE],
    }
    details = " ".join(f"{label}: {', '.join(ids)}." for label, ids in reasons.items() if ids)
    if decision is Decision.NEEDS_DOCUMENTATION:
        return f"Needs documentation before submission. {details}"
    return f"Escalated to a human reviewer. {details or 'The findings do not support a decision.'}"


def prompt_fingerprint(version: str) -> str:
    return hashlib.sha256(load_prompt(version).encode()).hexdigest()[:16]


def run_case(
    client: ModelClient,
    policy: Policy,
    request: PARequest,
    model: str = DEFAULT_MODEL,
    max_steps: int = MAX_STEPS,
    pipeline: Pipeline = Pipeline.SINGLE_AGENT,
) -> AuditRecord:
    """Match, verify and decide one case, and return its audit record. Both pipelines share verify and decide."""
    if pipeline is Pipeline.MULTI_AGENT:
        run = run_agents(client, policy, request, model=model, max_steps=max_steps)
    else:
        run = match(client, policy, request, model=model, max_steps=max_steps)
    return audit(run, policy, request, pipeline)


def determine(
    policy: Policy, request: PARequest, findings: list[CriterionResult], skipped: list[str]
) -> tuple[Determination, list[str]]:
    """Verify the findings and decide from the grounded ones. Returns the determination and every bad quote.

    Every path to a decision goes through here: both pipelines and the MCP server's check_findings.
    """
    checked = verify(request, findings)
    kept = {r.criterion_id for r in checked.grounded}
    rejected = [r.criterion_id for r in findings if r.criterion_id not in kept]
    decision = decide(policy, checked.grounded)
    determination = Determination(
        case_id=request.case_id,
        policy_id=policy.policy_id,
        decision=decision,
        criteria=checked.grounded,
        missing_documentation=missing_documentation(policy, checked.grounded),
        rationale=rationale(decision, policy, checked.grounded, skipped, rejected),
    )
    return determination, checked.problems


def audit(run: MatchRun, policy: Policy, request: PARequest, pipeline: Pipeline) -> AuditRecord:
    """Verify the run's findings, decide from the grounded ones, and record everything."""
    determination, problems = determine(policy, request, run.findings, run.skipped)
    return AuditRecord(
        determination=determination,
        policy_fingerprint=fingerprint(policy),
        model=run.model,
        prompt_version=run.prompt_version,
        prompt_fingerprint=prompt_fingerprint(run.prompt_version),
        stop=run.stop,
        steps=run.steps,
        usage=run.usage,
        skipped=run.skipped,
        rejected_quotes=problems,
        turns=run.turns,
        pipeline=pipeline,
    )
