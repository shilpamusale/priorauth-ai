"""The demo's data helpers, and the page itself rendered headlessly with Streamlit's test runner."""

import shutil
from pathlib import Path

import pytest
import yaml
from streamlit.testing.v1 import AppTest

import ui.data
from evals.gaps import run_gaps
from evals.gold import load_split
from evals.judge import REQUEST_CHECKS, run_judge
from evals.judge_items import ItemKind, build_items
from evals.runner import GOLD, RECORDINGS, Mode, record_trial, recording_path
from priorauth.llm.base import ModelResponse, StopReason, ToolCall, ToolResult, Usage
from priorauth.llm.scripted import ScriptedClient
from priorauth.matcher import DEFAULT_MODEL
from priorauth.pipeline import Pipeline
from priorauth.policies import load_policies, load_sources
from priorauth.retrieval import Candidate
from ui.data import SPLIT, brief, calibration, highlight, judge_grades, recorded_trials, replay, replay_all

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


@pytest.mark.parametrize(
    "content,error,shown",
    [
        (
            "n1 | 2026-03-10 | primary care visit\n\nLow back pain since March.",
            False,
            "n1 | 2026-03-10 | primary care visit",
        ),
        ("Recorded demo.duration. Still to submit: demo.scale.", False, "Recorded demo.duration"),
        (
            "quote 'ODI 48%' is not in n2. Copy each quote exactly.",
            True,
            "Error: quote 'ODI 48%' is not in n2. Copy each quote exactly.",
        ),
        ("", False, ""),
    ],
)
def test_brief_keeps_what_identifies_the_step(content, error, shown):
    assert brief(ToolResult(tool_call_id="toolu_1", content=content, is_error=error)) == shown


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
    usage = Usage(input_tokens=2000, output_tokens=200)
    client = ScriptedClient([ModelResponse(tool_calls=calls, stop_reason=StopReason.TOOL_USE, usage=usage)])
    path = recording_path(recordings, SPLIT, "dev_001", 1)
    record_trial(case, policy, client, path, DEFAULT_MODEL)
    return case, policy


def test_replay_scores_a_recorded_trial(real, tmp_path):
    policies, cases = real
    case, policy = record_dev_001(policies, cases, tmp_path)
    result = replay(case, policy, tmp_path, 1)
    assert result.record.determination.decision == "needs_documentation"
    assert {c.name: c.passed for c in result.checks}["decision_correct"] is False  # gold says ready


def test_replay_all_scores_every_recorded_trial(real, tmp_path):
    policies, cases = real
    record_dev_001(policies, cases, tmp_path)
    results = replay_all(policies, cases, tmp_path)
    assert list(results) == [("dev_001", 1)]
    assert results[("dev_001", 1)].record.determination.decision == "needs_documentation"


def test_replay_reports_a_missing_recording(real, tmp_path):
    policies, cases = real
    case = cases["dev_002"]
    assert replay(case, policies[case.policy_id], tmp_path, 1).error.startswith("no recording at")


class KeywordIndex:
    """Stands in for the embedding model on the page: an order lands on an LCD by its keywords.

    Like the real model, it does not know the shorthand "MBB", so dev_002 escalates at retrieval.
    """

    LCDS = {
        "lcdid=33944": ("eyelid", "blepharoplasty"),
        "lcdid=34082": ("vein", "saphenous", "phlebectomy"),
        "lcdid=38773": ("medial branch", "facet", "lumbar"),
    }

    def __init__(self, sources):
        self.sources = sorted(sources)

    def search(self, text):
        for key, words in self.LCDS.items():
            if any(word in text.lower() for word in words):
                source = next(s for s in self.sources if key in s)
                return [Candidate(source=source, distance=0.3)]
        return [Candidate(source=self.sources[0], distance=0.9)]


@pytest.fixture(autouse=True)
def keyword_index(monkeypatch):
    monkeypatch.setattr(ui.data, "build_index", lambda policy_dir: KeywordIndex(load_sources(policy_dir)))


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
    assert [(m.label, m.value) for m in app.metric] == [
        ("Runs with the expected decision", "0 of 1"),
        ("Hard-gate failures", "0"),
        ("Model cost per run", "$0.01"),
    ]
    policy = real[0]["l38773_first_mbb"]
    assert f"]({policy.source})" in app.markdown[0].value
    assert any('"Moderate to severe chronic neck or low back pain"' in t.value for t in app.text)
    log = [t.value for t in app.text]
    assert "Recorded l38773_first_mbb.pain_severity" in log
    assert not any("Still to submit" in line for line in log)  # the run log is shortened


def test_model_text_is_never_rendered_as_markdown(real, monkeypatch, tmp_path):
    """A model reply could carry markdown, such as an image that loads from another site; it is shown as text."""
    record_dev_001(*real, tmp_path)
    app = page(monkeypatch, tmp_path)
    assert any("![tracker](https://example.com/pixel.png)" in t.value for t in app.text)
    assert not any("example.com" in m.value for m in app.markdown)


class GapModel:
    """Answers each evidence agent with insufficient evidence for its one criterion, in any call order."""

    def complete(self, request):
        tool = next(t for t in request.tools if t.name == "submit_finding")
        [criterion_id] = tool.input_schema["properties"]["criterion_id"]["enum"]
        finding = {"criterion_id": criterion_id, "status": "insufficient_evidence", "evidence": [], "reasoning": "Gap."}
        call = ToolCall(id=f"toolu_{criterion_id}", name="submit_finding", input=finding)
        usage = Usage(input_tokens=1000, output_tokens=100)
        return ModelResponse(tool_calls=[call], stop_reason=StopReason.TOOL_USE, usage=usage)


def record_multi_dev_001(policies, cases, recordings):
    case = cases["dev_001"]
    path = recording_path(recordings / "multi_agent", SPLIT, "dev_001", 1)
    record_trial(case, policies[case.policy_id], GapModel(), path, DEFAULT_MODEL, Pipeline.MULTI_AGENT)


def test_the_page_switches_between_pipelines(real, monkeypatch, tmp_path):
    record_dev_001(*real, tmp_path)
    record_multi_dev_001(*real, tmp_path)
    app = page(monkeypatch, tmp_path)
    assert app.radio[0].options == ["Single agent (deployed)", "Multi-agent: one agent per criterion"]
    assert "Decision accuracy: single agent 0%, multi-agent 0% at" in app.caption[0].value
    app.radio[0].set_value(Pipeline.MULTI_AGENT).run()
    assert not app.exception
    assert [m.value for m in app.metric][2] == "$0.02"  # eight agents x (1,000 in at $2/M + 100 out at $10/M)
    assert sum(1 for e in app.expander if e.label.startswith("❔")) == 8


def test_replay_all_reads_each_pipelines_own_recordings(real, tmp_path):
    policies, cases = real
    record_multi_dev_001(policies, cases, tmp_path)
    assert replay_all(policies, cases, tmp_path) == {}
    multi = replay_all(policies, cases, tmp_path, Pipeline.MULTI_AGENT)
    assert multi[("dev_001", 1)].record.pipeline is Pipeline.MULTI_AGENT


def test_the_documentation_agents_requests_replace_the_gap_list(real, monkeypatch, tmp_path):
    policies, cases = real
    record_dev_001(policies, cases, tmp_path)  # pain severity met, the other seven criteria are gaps
    case = cases["dev_001"]
    gap_ids = [c.criterion_id for c in policies[case.policy_id].criteria][1:]
    written = [
        {
            "criterion_id": cid,
            "request": f"Please document {cid} clearly in the chart at the next visit.",
            "note_type": "visit note",
        }
        for cid in gap_ids
    ]
    calls = [ToolCall(id=f"toolu_{i}", name="write_request", input=w) for i, w in enumerate(written)]
    model = ScriptedClient([ModelResponse(tool_calls=calls, stop_reason=StopReason.TOOL_USE)])
    run_gaps(Mode.RECORD, policies, client=model, recordings=tmp_path, cases=["dev_001"])
    app = page(monkeypatch, tmp_path)
    assert any(m.value == "**Requests to the clinic before submitting**" for m in app.markdown)
    assert any(t.value.startswith("axial pain · visit note\nPlease document") for t in app.text)
    assert not any(m.value == "**Documentation to add before submitting**" for m in app.markdown)


def test_without_requests_the_gap_list_is_shown(real, monkeypatch, tmp_path):
    record_dev_001(*real, tmp_path)
    app = page(monkeypatch, tmp_path)
    assert any(m.value == "**Documentation to add before submitting**" for m in app.markdown)


class FakeJudge:
    """Grades evidence 'partly' with markdown in its reasoning, and fails requests on one check."""

    def complete(self, request):
        [tool] = request.tools
        if tool.name == "grade_evidence":
            args = {"grade": "partly", "reasoning": "Shown in part. ![tracker](https://example.com/judge.png)"}
        else:
            args = {**{name: name != "says_where" for name in REQUEST_CHECKS}, "reasoning": "No place named."}
        call = ToolCall(id="toolu_1", name=tool.name, input=args)
        return ModelResponse(
            tool_calls=[call], stop_reason=StopReason.TOOL_USE, usage=Usage(input_tokens=1, output_tokens=1)
        )


@pytest.fixture(scope="module")
def judged(tmp_path_factory):
    """The committed agent recordings, with made-up judge grades for dev_001 and dev_004 in place of the real ones,
    planted items included.

    The page tests never depend on the judge's committed recordings: the judge is not a merge gate.
    """
    recordings = tmp_path_factory.mktemp("recordings")
    shutil.copytree(RECORDINGS, recordings, dirs_exist_ok=True, ignore=shutil.ignore_patterns("judge"))
    items, cases = build_items(load_policies(Path("policies")), GOLD, recordings)
    chosen = [i for i in items if i.case_id in ("dev_001", "dev_004")]
    run_judge(Mode.RECORD, chosen, cases, client=FakeJudge(), recordings=recordings)
    return recordings


def test_judge_grades_are_keyed_by_kind_case_and_criterion(judged):
    grades = judge_grades(load_policies(Path("policies")), GOLD, judged)
    assert {case for _, case, _ in grades} == {"dev_001", "dev_004"}
    assert {kind for kind, _, _ in grades} == {ItemKind.EVIDENCE, ItemKind.REQUEST}  # planted items are not shown
    assert {g.grade for (kind, _, _), g in grades.items() if kind is ItemKind.EVIDENCE} == {"partly"}
    assert {g.grade for (kind, _, _), g in grades.items() if kind is ItemKind.REQUEST} == {"needs_revision"}


def test_no_judge_grades_without_the_runs_it_graded(real, tmp_path):
    record_dev_001(*real, tmp_path)  # the other dev cases have no first trial, so nothing to attach grades to
    assert judge_grades(real[0], GOLD, tmp_path) == {}


def test_calibration_is_none_without_a_report(tmp_path):
    assert calibration(tmp_path / "missing.json") is None
    assert calibration().evidence.items == 32


def test_the_judges_grade_shows_on_each_met_criterion_of_the_first_trial(judged, monkeypatch):
    app = page(monkeypatch, judged)
    assert app.selectbox[0].value == "dev_001" and app.selectbox[1].value == 1
    labels = [e.label for e in app.expander]
    assert any(label.startswith("✅ pain severity: met") and label.endswith(" · judge: partly") for label in labels)
    assert any(m.value == "**LLM judge: partly**" for m in app.markdown)
    report = calibration()
    agreed = f"agreed on {report.evidence.agreed} of {report.evidence.items} items"
    assert any(agreed in c.value and "pain duration can be wrong" in c.value for c in app.caption)


def test_the_judges_reasoning_is_never_rendered_as_markdown(judged, monkeypatch):
    app = page(monkeypatch, judged)
    assert any(t.value == "Shown in part. ![tracker](https://example.com/judge.png)" for t in app.text)
    assert not any("example.com" in m.value for m in [*app.markdown, *app.caption])


def test_other_trials_point_to_the_judged_one(judged, monkeypatch):
    app = page(monkeypatch, judged)
    app.selectbox[1].set_value(2).run()
    assert not any("judge:" in e.label for e in app.expander)
    assert any(c.value.startswith("Pick trial 1 of the single agent") for c in app.caption)
    app.selectbox[1].set_value(1).run()
    app.radio[0].set_value(Pipeline.MULTI_AGENT).run()
    assert not any("judge:" in e.label for e in app.expander)


def test_the_judges_grade_shows_under_each_request(judged, monkeypatch):
    app = page(monkeypatch, judged)
    app.selectbox[0].set_value("dev_004").run()
    shown = [t.value for t in app.text if "\nJudge: " in t.value]
    assert shown and all(t.endswith("\nJudge: needs revision (fails: says where)") for t in shown)


def test_every_dev_request_is_routed_in_a_table(monkeypatch):
    app = page(monkeypatch, RECORDINGS)
    [expander] = [e for e in app.expander if e.label.startswith("How a request finds its policy")]
    rows = app.dataframe[0].value
    assert len(rows) == len(list((GOLD / "dev").glob("*.yaml"))) + len(
        yaml.safe_load((GOLD / "routing" / "dev.yaml").read_text())
    )
    assert any(f"Dev requests: {len(rows) - 1} of {len(rows)} routed right; 0 reached" in c.value for c in app.caption)
    assert list(rows[rows["case"] == "dev_002"]["right"]) == ["no, escalated"]


def test_the_selected_case_shows_its_order_and_route(monkeypatch):
    app = page(monkeypatch, RECORDINGS)
    route_text = [t.value for t in app.text if t.value.startswith("Order: ")]
    assert route_text == [
        "Order: Diagnostic lumbar medial branch blocks, L4-L5 and L5-S1\nRetrieval: The order matches "
        "LCD L38773 v32 and l38773_first_mbb covers code 64493."
    ]
    assert not any("Retrieval would send this request" in w.value for w in app.warning)


def test_a_case_retrieval_would_escalate_says_so(monkeypatch):
    app = page(monkeypatch, RECORDINGS)
    app.selectbox[0].set_value("dev_002").run()
    assert any(w.value.startswith("Retrieval would send this request to a person") for w in app.warning)


def try_route(app, order, code):
    app.text_input[0].set_value(order)
    app.text_input[1].set_value(code)
    app.button[0].click().run()
    return app


def test_a_visitor_can_try_retrieval(monkeypatch):
    app = try_route(page(monkeypatch, RECORDINGS), "Upper eyelid blepharoplasty, both eyes", "15823")
    [answer] = [t.value for t in app.text if t.value.startswith("matched")]
    assert answer.startswith("matched: l33944_upper_bleph\nThe order matches LCD L33944 v28")


def test_a_tried_order_is_never_rendered_as_markdown(monkeypatch):
    app = try_route(page(monkeypatch, RECORDINGS), "![x](https://example.com/a.png) **knee** replacement", "27447")
    assert any(t.value.startswith("no_match\nNo coverage document is close enough") for t in app.text)
    assert not any("example.com" in m.value for m in [*app.markdown, *app.caption])


def test_a_malformed_service_code_is_refused(monkeypatch):
    app = try_route(page(monkeypatch, RECORDINGS), "Upper eyelid blepharoplasty", "123")
    assert app.error and "five characters" in app.error[0].value


def test_route_rows_mark_an_unsafe_route_apart_from_a_safe_miss():
    from evals.routing import RoutingCase, RoutingResult
    from priorauth.retrieval import Outcome, Route
    from ui.data import route_rows

    case = RoutingCase(case_id="c", ordered_service="Knee", service_code="64493", expected="no_match")
    reached = Route(outcome=Outcome.MATCHED, policy_id="l38773_first_mbb", reason="r")
    escalated = Route(outcome=Outcome.UNSUPPORTED, reason="r")
    rows = route_rows([RoutingResult(case=case, route=reached), RoutingResult(case=case, route=escalated)])
    assert [row["right"] for row in rows] == ["UNSAFE", "no, escalated"]
    assert rows[0]["closest LCD"] == "" and rows[0]["distance"] is None
