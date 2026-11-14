"""The demo's data helpers, and the page itself rendered headlessly with Streamlit's test runner."""

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from evals.gold import load_split
from evals.runner import record_trial, recording_path
from priorauth.llm.base import ModelResponse, StopReason, ToolCall
from priorauth.llm.scripted import ScriptedClient
from priorauth.matcher import DEFAULT_MODEL
from priorauth.policies import load_policies
from ui.data import SPLIT, highlight, recorded_trials, replay

APP = Path(__file__).resolve().parents[1] / "ui" / "app.py"


def test_highlight_marks_each_quote():
    assert highlight("Pain NRS 7/10. ODI 44%.", ["NRS 7/10"]) == "Pain <mark>NRS 7/10</mark>. ODI 44%."


def test_highlight_escapes_the_note():
    """A note is data: markup inside it is shown as text, never rendered."""
    shown = highlight('<script>alert("x")</script> ODI 44% & rising', ["ODI 44% & rising"])
    assert "<script>" not in shown
    assert shown == "&lt;script&gt;alert(&quot;x&quot;)&lt;/script&gt; <mark>ODI 44% &amp; rising</mark>"


def test_highlight_matches_like_verify_does():
    """Curly quotes and extra spaces are normalized on both sides, in the note and in the quote."""
    marked = "<mark>the &quot;patient&quot; reports</mark>"
    assert highlight("the \u201cpatient\u201d  reports", ['the "patient" reports']) == marked
    assert highlight('the "patient" reports', ["the  \u201cpatient\u201d reports"]) == marked


def test_overlapping_quotes_become_one_mark():
    assert highlight("Failed naproxen for 3 months.", ["Failed naproxen", "naproxen for 3 months"]) == (
        "<mark>Failed naproxen for 3 months</mark>."
    )


def test_no_quotes_leaves_the_note_unmarked():
    assert highlight("No leg pain.", []) == "No leg pain."


def test_recorded_trials_sorts_numerically_and_skips_cases_without_recordings(tmp_path):
    for trial in [1, 2, 10]:
        path = tmp_path / SPLIT / "dev_001" / f"trial_{trial}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}")
    assert recorded_trials(tmp_path, ["dev_001", "dev_002"]) == {"dev_001": [1, 2, 10]}


@pytest.fixture
def real():
    policies = load_policies(Path("policies"))
    return policies, load_split(Path("evals/gold"), SPLIT, policies)


def record_dev_001(policies, cases, recordings):
    """A scripted run of dev_001: one criterion met with a real quote, the rest left without documentation."""
    case = cases["dev_001"]
    policy = policies[case.policy_id]
    findings = [
        {
            "criterion_id": c.criterion_id,
            "status": "insufficient_evidence",
            "evidence": [],
            "reasoning": "Not reviewed.",
        }
        for c in policy.criteria
    ]
    findings[0] = {
        "criterion_id": policy.criteria[0].criterion_id,
        "status": "met",
        "evidence": [{"note_id": "n4", "quote": "Pain NRS 7/10"}],
        "reasoning": "NRS 7/10 at the consult. ![tracker](https://example.com/pixel.png)",
    }
    calls = [ToolCall(id=f"toolu_{i}", name="submit_finding", input=f) for i, f in enumerate(findings)]
    client = ScriptedClient([ModelResponse(tool_calls=calls, stop_reason=StopReason.TOOL_USE)])
    path = recording_path(recordings, SPLIT, "dev_001", 1)
    record_trial(case, policy, client, path, DEFAULT_MODEL)
    return case, policy


def test_replay_scores_a_recorded_trial(real, tmp_path):
    policies, cases = real
    case, policy = record_dev_001(policies, cases, tmp_path)
    result = replay(case, policy, tmp_path, 1)
    assert result.record.determination.decision == "needs_documentation"
    assert {c.name: c.passed for c in result.checks}["decision_correct"] is False  # gold says ready


def test_replay_reports_a_missing_recording(real, tmp_path):
    policies, cases = real
    case = cases["dev_002"]
    assert replay(case, policies[case.policy_id], tmp_path, 1).error.startswith("no recording at")


def page(monkeypatch, recordings):
    monkeypatch.setenv("PRIORAUTH_RECORDINGS", str(recordings))
    app = AppTest.from_file(str(APP), default_timeout=60)
    app.run()
    assert not app.exception, app.exception
    return app


def test_page_without_recordings_says_how_to_make_one(monkeypatch, tmp_path):
    app = page(monkeypatch, tmp_path)
    assert "No recorded runs yet" in app.info[0].value


def test_page_shows_a_recorded_trial(real, monkeypatch, tmp_path):
    record_dev_001(*real, tmp_path)
    app = page(monkeypatch, tmp_path)
    assert app.selectbox[0].value == "dev_001"
    assert app.warning[0].value.startswith("Needs documentation before submission.")
    text = " ".join(m.value for m in app.markdown)
    assert any("<mark>Pain NRS 7/10</mark>" in h.proto.body for h in app.get("html"))
    assert "Expected **ready_to_submit**; this trial differs from it." in text
    assert any(e.label.startswith("✅ pain severity: met (matches the gold label)") for e in app.expander)


def test_model_text_is_never_rendered_as_markdown(real, monkeypatch, tmp_path):
    """A model reply could carry markdown, such as an image that loads from another site; it is shown as text."""
    record_dev_001(*real, tmp_path)
    app = page(monkeypatch, tmp_path)
    assert any("![tracker](https://example.com/pixel.png)" in t.value for t in app.text)
    assert not any("example.com" in m.value for m in app.markdown)
