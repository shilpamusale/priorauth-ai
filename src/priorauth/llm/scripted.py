"""A model client that plays back pre-written responses, so tests exercis the agent without an API Key."""

from priorauth.llm.base import ModelRequest, ModelResponse


class ScriptedClient:
    """Returns its responses in order and keeps a copy of every request it was sent."""

    def __init__(self, responses: list[ModelResponse]) -> None:
        self._responses = list(responses)
        self.requests: list[ModelRequest] = []

    def complete(self, request:ModelRequest) -> ModelResponse:
        # A deep copy, so later changes to the conversation cannot rewrite what was sent.
        self.requests.append(request.model_copy(deep=True))
        if not self._responses:
            raise RuntimeError("scripted client has no responses left")
        return self._responses.pop(0)
    
    