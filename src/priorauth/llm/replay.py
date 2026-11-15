"""Record live model calls and replay them, so CI can score the agent without an API key.

A recording is the list of exchanges from one run: a fingerprint of each request and the reply it got.
Replay is strict: a reply is returned only for the exact request it was recorded for, so a change to the
prompt, the tools, the model or the chart makes the recording stale instead of quietly replaying answers
to a different question. Replies are found by fingerprint, not position, so agents running in parallel
replay correctly whatever order their calls arrive in. Each request already contains the whole
conversation before it, so a single agent's run is just as strictly checked as when replay went by order.
"""

import hashlib
import json
import threading
from pathlib import Path

from priorauth.llm.base import ModelClient, ModelRequest, ModelResponse
from priorauth.schemas import Contract


class StaleRecording(Exception):
    """The run no longer matches its recording; record the case again."""


class Exchange(Contract):
    request_fingerprint: str
    response: ModelResponse


class Recording(Contract):
    exchanges: list[Exchange] = []


def request_fingerprint(request: ModelRequest) -> str:
    """Short hash of everything sent to the model. Keys are sorted, so dict order does not matter."""
    canonical = json.dumps(request.model_dump(mode="json"), sort_keys=True)
    return hashlib.sha256(canonical.encode()).hexdigest()[:16]


def save_recording(recording: Recording, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(recording.model_dump_json(indent=1) + "\n", encoding="utf-8")


def load_recording(path: Path) -> Recording:
    return Recording.model_validate_json(path.read_text(encoding="utf-8"))


class RecordingClient:
    """Passes each call to a live client and keeps the exchange. Safe to share between threads."""

    def __init__(self, client: ModelClient) -> None:
        self._client = client
        self.recording = Recording()
        self._lock = threading.Lock()

    def complete(self, request: ModelRequest) -> ModelResponse:
        response = self._client.complete(request)
        with self._lock:
            self.recording.exchanges.append(
                Exchange(request_fingerprint=request_fingerprint(request), response=response)
            )
        return response


class ReplayClient:
    """Returns the recorded reply for each request it was recorded for, and refuses any other request.

    Safe to share between threads. If the same request was recorded more than once, its replies come back
    in the order they were recorded.
    """

    def __init__(self, recording: Recording) -> None:
        self._replies: dict[str, list[ModelResponse]] = {}
        for exchange in recording.exchanges:
            self._replies.setdefault(exchange.request_fingerprint, []).append(exchange.response)
        self._recorded = len(recording.exchanges)
        self._used = 0
        self._lock = threading.Lock()

    def complete(self, request: ModelRequest) -> ModelResponse:
        key = request_fingerprint(request)
        with self._lock:
            if self._used == self._recorded:
                raise StaleRecording(f"the run made more than the {self._recorded} recorded model calls")
            if not self._replies.get(key):
                raise StaleRecording(f"model call {self._used + 1} differs from the recording")
            self._used += 1
            return self._replies[key].pop(0)

    def check_finished(self) -> None:
        """A run that stops before using every recorded reply has changed too."""
        if self._used != self._recorded:
            raise StaleRecording(f"the run made {self._used} of the {self._recorded} recorded model calls")
