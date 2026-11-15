"""Multi-agent matching: a code orchestrator runs one evidence agent per criterion, in parallel.

Each evidence agent is the matcher given a one-criterion view of the policy: the same prompt and tools, but
it can submit only its own criterion. Comparing this with the single agent therefore measures the
orchestration, not a prompt change. The orchestrator is plain code, not a model: it starts the agents,
collects their findings in policy order, and the pipeline hands them to the same verify() and decide().
"""

from concurrent.futures import ThreadPoolExecutor

from priorauth.llm.base import ModelClient, Usage
from priorauth.matcher import DEFAULT_MODEL, MAX_STEPS, MatchRun, StopCause, match
from priorauth.schemas import PARequest, Policy

WORKERS = 8  # agents running at once


def combine(runs: list[MatchRun]) -> MatchRun:
    """The agents' runs as one, in policy order. The first agent that did not complete sets the stop cause."""
    stops = [run.stop for run in runs if run.stop is not StopCause.COMPLETE]
    return MatchRun(
        model=runs[0].model,
        prompt_version=runs[0].prompt_version,
        findings=[finding for run in runs for finding in run.findings],
        skipped=[criterion_id for run in runs for criterion_id in run.skipped],
        stop=stops[0] if stops else StopCause.COMPLETE,
        steps=sum(run.steps for run in runs),
        usage=Usage(
            input_tokens=sum(run.usage.input_tokens for run in runs),
            output_tokens=sum(run.usage.output_tokens for run in runs),
        ),
        turns=[turn for run in runs for turn in run.turns],
    )


def run_agents(
    client: ModelClient,
    policy: Policy,
    request: PARequest,
    model: str = DEFAULT_MODEL,
    max_steps: int = MAX_STEPS,
    workers: int = WORKERS,
) -> MatchRun:
    """One evidence agent per criterion, in parallel; the client must be safe to share between threads."""
    views = [policy.model_copy(update={"criteria": [criterion]}) for criterion in policy.criteria]
    with ThreadPoolExecutor(max_workers=workers) as pool:
        runs = list(pool.map(lambda view: match(client, view, request, model=model, max_steps=max_steps), views))
    return combine(runs)  # pool.map keeps the input order, so this is policy order
