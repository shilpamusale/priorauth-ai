"""HTTP API for the pipeline: send a prior authorization request, get back the determination and its audit record.

POST /requests finds the policy with retrieval from the order text, then runs the pipeline on it; a request
that reaches no policy escalates with the reason and no model call. POST /determinations takes a named
policy instead.

POST /determinations calls the live model, so the service is deployed privately (only callers with Google
Cloud access can reach it) and a daily limit caps the number of runs. The Anthropic workspace's monthly
spend limit is the hard backstop. The public demo never calls this API; it replays recordings.

    uvicorn priorauth.api:app --port 8000      # needs the api and agent extras and ANTHROPIC_API_KEY
"""

import datetime
import os
import threading
from collections.abc import Callable
from functools import cache
from pathlib import Path

from fastapi import FastAPI, HTTPException

from priorauth.llm.base import ModelClient
from priorauth.matcher import DEFAULT_MODEL
from priorauth.pipeline import AuditRecord, run_case
from priorauth.policies import fingerprint, load_policies, load_sources
from priorauth.retrieval import Outcome, PolicyIndex, Route, route
from priorauth.schemas import Contract, PARequest

DAILY_RUNS = 50  # at about 3 cents a run, at most about $1.50 a day


class DeterminationRequest(Contract):
    policy_id: str
    request: PARequest


class RoutedRequest(Contract):
    """The route retrieval found, and the audit record when the request reached a policy."""

    route: Route
    audit: AuditRecord | None = None


class PolicySummary(Contract):
    policy_id: str
    title: str
    source: str
    fingerprint: str
    criteria: list[str]


class DailyLimit:
    """Counts runs per UTC day. In memory, so it is per server process: the service runs one instance."""

    def __init__(self, limit: int, today: Callable[[], datetime.date]) -> None:
        self.limit = limit
        self._today = today
        self._day: datetime.date | None = None
        self._used = 0
        self._lock = threading.Lock()

    def take(self) -> bool:
        """Reserve one run; False when today's runs are used up. A run counts even if it later fails."""
        with self._lock:
            if self._day != self._today():
                self._day, self._used = self._today(), 0
            if self._used >= self.limit:
                return False
            self._used += 1
            return True


def utc_today() -> datetime.date:
    return datetime.datetime.now(datetime.UTC).date()


def seconds_until_tomorrow() -> int:
    now = datetime.datetime.now(datetime.UTC)
    tomorrow = datetime.datetime.combine(now.date() + datetime.timedelta(days=1), datetime.time(), datetime.UTC)
    return max(int((tomorrow - now).total_seconds()), 1)


def claude_client() -> ModelClient:
    from priorauth.llm.claude import ClaudeClient  # needs the agent extra and ANTHROPIC_API_KEY

    return ClaudeClient()


def create_app(
    policy_dir: Path | None = None,
    daily_runs: int | None = None,
    client_factory: Callable[[], ModelClient] = claude_client,
    today: Callable[[], datetime.date] = utc_today,
    index_factory: Callable[[dict[str, str]], PolicyIndex] = PolicyIndex,
) -> FastAPI:
    """The service. Policies load (and their LCD quotes are checked) at startup, so a bad policy fails fast.

    Without arguments, the policy folder and the daily limit come from PRIORAUTH_POLICIES and
    PRIORAUTH_DAILY_RUNS, so the deployment can change them without a code change. The retrieval index is
    built on the first request that needs it.
    """
    folder = policy_dir or Path(os.environ.get("PRIORAUTH_POLICIES", "policies"))
    policies = load_policies(folder)
    if daily_runs is None:
        daily_runs = int(os.environ.get("PRIORAUTH_DAILY_RUNS", DAILY_RUNS))
    limit = DailyLimit(daily_runs, today)
    client = cache(client_factory)  # one model client, created on the first run
    index = cache(lambda: index_factory(load_sources(folder)))  # one index, built on the first routed request
    app = FastAPI(title="PriorAuth-AI", summary="Is this prior authorization request ready to submit?")

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/policies")
    def list_policies() -> list[PolicySummary]:
        return [
            PolicySummary(
                policy_id=p.policy_id,
                title=p.title,
                source=p.source,
                fingerprint=fingerprint(p),
                criteria=[c.criterion_id for c in p.criteria],
            )
            for p in policies.values()
        ]

    def take_run() -> None:
        if not limit.take():
            headers = {"Retry-After": str(seconds_until_tomorrow())}
            raise HTTPException(status_code=429, detail="daily run limit reached", headers=headers)

    @app.post("/requests")
    def take_request(request: PARequest) -> RoutedRequest:
        """Find the policy from the order text, then run the pipeline on it with the live model.

        A request that reaches no policy comes back at once with the reason, and uses no run.
        """
        found = route(index(), policies.values(), request.ordered_service, request.service_code)
        if found.outcome is not Outcome.MATCHED:
            return RoutedRequest(route=found)
        take_run()
        audit = run_case(client(), policies[found.policy_id], request, model=DEFAULT_MODEL)
        return RoutedRequest(route=found, audit=audit)

    @app.post("/determinations")
    def determine(body: DeterminationRequest) -> AuditRecord:
        """Run the pipeline on one request against a named policy, with the live model. Takes up to a minute."""
        if body.policy_id not in policies:
            raise HTTPException(status_code=404, detail=f"unknown policy {body.policy_id!r}")
        policy = policies[body.policy_id]
        if body.request.service_code not in policy.service_codes:
            detail = f"{policy.policy_id} does not cover code {body.request.service_code}"
            raise HTTPException(status_code=422, detail=detail)
        take_run()
        return run_case(client(), policy, body.request, model=DEFAULT_MODEL)

    return app


app = create_app()
