"""The tools the matcher uses to read a chart and record its findings.

The prompt lists the notes (ID, date, type) but not their text. The model reads a note with read_note,
finds passages across the chart with search_chart, and records one finding per criterion with
submit_finding. A bad call comes back as an error result the model can correct; it never raises.
"""

import re
from typing import Any

from pydantic import ValidationError

from priorauth.ingest import normalize
from priorauth.llm.base import ToolCall, ToolResult, ToolSpec
from priorauth.schemas import (
    ClinicalNote,
    Contract,
    CriterionResult,
    CriterionStatus,
    NonBlank,
    NoteId,
    PARequest,
    Policy,
)

MIN_QUOTE_WORDS = 2  # one word ("pain", "PT") cannot show a criterion is met; two can ("ODI 44%", "NRS 7/10")
MAX_SEARCH_HITS = 10
SNIPPET_CHARS = 80  # text shown on each side of a search match


class ToolError(Exception):
    """A tool call the model can fix; its message is returned to the model as an error result."""


class ReadNoteInput(Contract):
    note_id: NoteId


class SearchChartInput(Contract):
    query: NonBlank


def strict_object(properties: dict[str, Any]) -> dict[str, Any]:
    """A JSON Schema object in the form strict tool use expects: every property required, no others allowed."""
    return {"type": "object", "properties": properties, "required": list(properties), "additionalProperties": False}


def describe(error: ValidationError) -> str:
    """A validation error as one line the model can act on."""
    return "; ".join(f"{'.'.join(map(str, e['loc'])) or 'input'}: {e['msg']}" for e in error.errors())


class ChartTools:
    """The policy and chart for one case, the tools over them, and the findings submitted so far."""

    def __init__(self, policy: Policy, request: PARequest) -> None:
        self.policy = policy
        self.notes = {note.note_id: note for note in request.notes}
        self.findings: dict[str, CriterionResult] = {}

    def note_index(self) -> str:
        """One line per note for the prompt: ID, date and type, without the text."""
        return "\n".join(f"{note.note_id} | {note.date} | {note.note_type}" for note in self.notes.values())

    def remaining(self) -> list[str]:
        """Criteria with no finding yet, in policy order."""
        return [c.criterion_id for c in self.policy.criteria if c.criterion_id not in self.findings]

    def specs(self) -> list[ToolSpec]:
        """Tool definitions for this case. Criterion and note IDs are enums, so strict tool use binds the
        model's input to this policy and this chart."""
        note_id = {"type": "string", "enum": list(self.notes)}
        return [
            ToolSpec(
                name="read_note",
                description="Return the full text of one note in the chart.",
                input_schema=strict_object({"note_id": note_id}),
            ),
            ToolSpec(
                name="search_chart",
                description=(
                    "Find a word or phrase in every note, ignoring case. "
                    "Returns each match with its note ID and the surrounding text."
                ),
                input_schema=strict_object({"query": {"type": "string"}}),
            ),
            ToolSpec(
                name="submit_finding",
                description=(
                    "Record your finding for one policy criterion. Each criterion is submitted once and cannot be "
                    "changed. Every evidence quote must be copied exactly from the note it names, and a criterion "
                    "marked met must cite at least one quote."
                ),
                input_schema=strict_object(
                    {
                        "criterion_id": {"type": "string", "enum": [c.criterion_id for c in self.policy.criteria]},
                        "status": {"type": "string", "enum": [status.value for status in CriterionStatus]},
                        "evidence": {
                            "type": "array",
                            "items": strict_object({"note_id": note_id, "quote": {"type": "string"}}),
                        },
                        "reasoning": {"type": "string"},
                    }
                ),
            ),
        ]

    def run(self, call: ToolCall) -> ToolResult:
        """Run one tool call. Anything wrong with the call comes back as an error result."""
        handlers = {
            "read_note": self._read_note,
            "search_chart": self._search_chart,
            "submit_finding": self._submit_finding,
        }
        try:
            if call.name not in handlers:
                raise ToolError(f"unknown tool {call.name!r}")
            return ToolResult(tool_call_id=call.id, content=handlers[call.name](call.input))
        except ValidationError as error:
            message = describe(error)
        except ToolError as error:
            message = str(error)
        return ToolResult(tool_call_id=call.id, content=message, is_error=True)

    def _note(self, note_id: str) -> ClinicalNote:
        if note_id not in self.notes:
            raise ToolError(f"no note {note_id!r} in this chart; the notes are {', '.join(self.notes)}")
        return self.notes[note_id]

    def _read_note(self, args: dict[str, Any]) -> str:
        note = self._note(ReadNoteInput.model_validate(args).note_id)
        return f"{note.note_id} | {note.date} | {note.note_type}\n\n{note.text}"

    def _search_chart(self, args: dict[str, Any]) -> str:
        query = normalize(SearchChartInput.model_validate(args).query)
        pattern = re.compile(re.escape(query), re.IGNORECASE)
        hits = []
        for note in self.notes.values():
            text = normalize(note.text)
            for match in pattern.finditer(text):
                start, end = max(0, match.start() - SNIPPET_CHARS), match.end() + SNIPPET_CHARS
                hits.append(f"{note.note_id}: {'...' * (start > 0)}{text[start:end]}{'...' * (end < len(text))}")
        if not hits:
            return f"no matches for {query!r}"
        shown = "\n".join(hits[:MAX_SEARCH_HITS])
        return shown if len(hits) <= MAX_SEARCH_HITS else f"{shown}\n(first {MAX_SEARCH_HITS} of {len(hits)} matches)"

    def _submit_finding(self, args: dict[str, Any]) -> str:
        finding = CriterionResult.model_validate(args)
        if finding.criterion_id not in {c.criterion_id for c in self.policy.criteria}:
            raise ToolError(f"{finding.criterion_id!r} is not a criterion of this policy")
        if finding.criterion_id in self.findings:
            raise ToolError(f"{finding.criterion_id} already has a finding, and findings cannot be changed")
        problems = []
        for item in finding.evidence:
            if len(item.quote.split()) < MIN_QUOTE_WORDS:
                problems.append(f"quote {item.quote!r} is too short; quote at least {MIN_QUOTE_WORDS} words")
            elif normalize(item.quote) not in normalize(self._note(item.note_id).text):
                problems.append(f"quote {item.quote!r} is not in {item.note_id}; copy the text exactly from the note")
        if problems:
            raise ToolError("; ".join(problems))
        self.findings[finding.criterion_id] = finding
        left = self.remaining()
        return f"Recorded {finding.criterion_id}. " + (
            f"Still to submit: {', '.join(left)}." if left else "All criteria are submitted."
        )