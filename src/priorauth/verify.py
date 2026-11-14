"""Final grounding check: every quote a finding cites must appear in the note it names.

submit_finding runs the same check so the model can fix a bad quote while it works. This runs it again
on the finished results, whatever produced them, and only grounded findings reach decide(). A finding
with any bad quote is dropped whole, so a required criterion it covered counts as unchecked and the
case escalates to a human.
"""

from collections.abc import Mapping

from priorauth.ingest import normalize
from priorauth.schemas import ClinicalNote, Contract, CriterionResult, Evidence, PARequest

MIN_QUOTE_WORDS = 2  # one word ("pain", "PT") cannot show a criterion is met; two can ("ODI 44%", "NRS 7/10")


class Verification(Contract):
    grounded: list[CriterionResult]  # findings whose every quote is in the note it names
    problems: list[str] = []  # one line per bad quote, for the audit record


def quote_problem(evidence: Evidence, notes: Mapping[str, ClinicalNote]) -> str | None:
    """Why this quote does not ground a finding, or None if it does."""
    if evidence.note_id not in notes:
        return f"no note {evidence.note_id!r} in this chart"
    if len(evidence.quote.split()) < MIN_QUOTE_WORDS:
        return f"quote {evidence.quote!r} is too short; quote at least {MIN_QUOTE_WORDS} words"
    if normalize(evidence.quote) not in normalize(notes[evidence.note_id].text):
        return f"quote {evidence.quote!r} is not in {evidence.note_id}"
    return None


def verify(request: PARequest, results: list[CriterionResult]) -> Verification:
    """Keep the findings whose quotes are all grounded in the chart; report every bad quote."""
    notes = {note.note_id: note for note in request.notes}
    grounded, problems = [], []
    for result in results:
        found = [problem for item in result.evidence if (problem := quote_problem(item, notes))]
        problems += [f"{result.criterion_id}: {problem}" for problem in found]
        if not found:
            grounded.append(result)
    return Verification(grounded=grounded, problems=problems)