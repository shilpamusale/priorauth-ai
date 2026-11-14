"""Record live model calls and replay them, so CI can score the agent without an API key.

A recording is the ordered list of exchanges from one run: a fingerprint of each request and the reply it
got. Replay is strict. Each request must match the fingerprint recorded at that position, so a change to
the prompt, the tools, the model or the chart makes the recording stale instead of quietly replaying
answers to a different question.
"""

import hashlib
import json
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
    """Passes each call to a live client and keeps the exchange."""

    def __init__(self, client: ModelClient) -> None:
        self._client = client
        self.recording = Recording()

    def complete(self, request: ModelRequest) -> ModelResponse:
        response = self._client.complete(request)
        self.recording.exchanges.append(Exchange(request_fingerprint=request_fingerprint(request), response=response))
        return response


class ReplayClient:
    """Returns the recorded replies in order, but only for the requests they were recorded for."""

    def __init__(self, recording: Recording) -> None:
        self._exchanges = list(recording.exchanges)
        self._next = 0

    def complete(self, request: ModelRequest) -> ModelResponse:
        if self._next == len(self._exchanges):
            raise StaleRecording(f"the run made more than the {len(self._exchanges)} recorded model calls")
        exchange = self._exchanges[self._next]
        if request_fingerprint(request) != exchange.request_fingerprint:
            raise StaleRecording(f"model call {self._next + 1} differs from the recording")
        self._next += 1
        return exchange.response

    def check_finished(self) -> None:
        """A run that stops before using every recorded reply has changed too."""
        if self._next != len(self._exchanges):
            raise StaleRecording(f"the run made {self._next} of the {len(self._exchanges)} recorded model calls")