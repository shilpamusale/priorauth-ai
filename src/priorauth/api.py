"""HTTP API for the pipeline: send a prior authorization request, get back the determination and its audit record.

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
from priorauth.policies import fingerprint, load_policies
from priorauth.schemas import Contract, PARequest

DAILY_RUNS = 50  # at about 3 cents a run, at most about $1.50 a day


class DeterminationRequest(Contract):
    policy_id: str
    request: PARequest


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
) -> FastAPI:
    """The service. Policies load (and their LCD quotes are checked) at startup, so a bad policy fails fast.

    Without arguments, the policy folder and the daily limit come from PRIORAUTH_POLICIES and
    PRIORAUTH_DAILY_RUNS, so the deployment can change them without a code change.
    """
    policies = load_policies(policy_dir or Path(os.environ.get("PRIORAUTH_POLICIES", "policies")))
    if daily_runs is None:
        daily_runs = int(os.environ.get("PRIORAUTH_DAILY_RUNS", DAILY_RUNS))
    limit = DailyLimit(daily_runs, today)
    client = cache(client_factory)  # one model client, created on the first run
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

    @app.post("/determinations")
    def determine(body: DeterminationRequest) -> AuditRecord:
        """Run the pipeline on one request with the live model. Takes up to a minute."""
        if body.policy_id not in policies:
            raise HTTPException(status_code=404, detail=f"unknown policy {body.policy_id!r}")
        if not limit.take():
            headers = {"Retry-After": str(seconds_until_tomorrow())}
            raise HTTPException(status_code=429, detail="daily run limit reached", headers=headers)
        return run_case(client(), policies[body.policy_id], body.request, model=DEFAULT_MODEL)

    return app


app = create_app()
