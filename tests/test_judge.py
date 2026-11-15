"""The LLM judge, with scripted and rule-based fake models: no API key, no network."""

import json
import shutil
from pathlib import Path

import pytest

from evals.judge import (
    JUDGE_MODEL,
    REQUEST_CHECKS,
    GradeTool,
    cost_usd,
    item_message,
    judge_item,
    judge_path,
    judged_items,
    main,
    request_grade,
    run_judge,
)
from evals.judge_items import ItemKind, build_items, load_sample
from evals.runner import Mode
from priorauth.llm.base import ModelResponse, StopReason, ToolCall, Usage
from priorauth.llm.scripted import ScriptedClient
from priorauth.policies import load_policies


@pytest.fixture(scope="module")
def built():
    return build_items(load_policies(Path("policies")))


def first(items, kind):
    return next(i for i in items if i.kind is kind)


def reply(name, **args):
    call = ToolCall(id="toolu_1", name=name, input=args)
    return ModelResponse(
        tool_calls=[call], stop_reason=StopReason.TOOL_USE, usage=Usage(input_tokens=1300, output_tokens=500)
    )


class RuleJudge:
    """Grades by what it is shown: evidence always 'partly', requests fail one check. Order-independent."""

    def complete(self, request):
        if "<evidence_item>" in request.turns[0].text:
            return reply("grade_evidence", grade="partly", reasoning="Only part is shown.")
        checks = {name: name != "says_where" for name in REQUEST_CHECKS}
        return reply("grade_request", **checks, reasoning="It does not say where to record it.")


def test_the_judge_grades_115_items_every_evidence_item_request_and_sampled_planted_item(built):
    items, _ = built
    chosen = judged_items(items, load_sample())
    kinds = [i.kind for i in chosen]
    assert (kinds.count(ItemKind.EVIDENCE), kinds.count(ItemKind.PLANTED), kinds.count(ItemKind.REQUEST)) == (99, 8, 8)
    assert {i.item_id for i in chosen if i.kind is ItemKind.PLANTED} <= set(load_sample())


def test_an_evidence_item_shows_the_claim_and_cited_notes_but_not_who_made_it(built):
    items, cases = built
    item = first(items, ItemKind.EVIDENCE)
    message = item_message(item, cases[item.case_id])
    assert item.criterion_text in message and "The criterion is met, citing:" in message
    assert all(q.quote in message for q in item.quotes)
    cited = {q.note_id for q in item.quotes}
    uncited = [n for n in cases[item.case_id].request.notes if n.note_id not in cited]
    assert uncited and not any(f'id="{n.note_id}"' in message for n in uncited)
    assert "planted" not in message and "reasoning" not in message.lower()


def test_a_planted_item_reads_exactly_like_an_evidence_item(built):
    items, cases = built
    planted = first(items, ItemKind.PLANTED)
    message = item_message(planted, cases[planted.case_id])
    assert message.startswith("<evidence_item>") and "planted" not in message


def test_a_request_item_shows_the_request_and_the_whole_chart(built):
    items, cases = built
    item = first(items, ItemKind.REQUEST)
    message = item_message(item, cases[item.case_id])
    assert item.request in message and f'note_type="{item.note_type}"' in message
    assert all(f'id="{n.note_id}"' in message for n in cases[item.case_id].request.notes)


def test_evidence_grade_is_the_judges_choice(built):
    items, cases = built
    item = first(items, ItemKind.EVIDENCE)
    model = ScriptedClient([reply("grade_evidence", grade="does_not_support", reasoning="Wrong criterion.")])
    judgement = judge_item(model, item, cases[item.case_id])
    assert (judgement.grade.grade, judgement.grade.reasoning) == ("does_not_support", "Wrong criterion.")
    assert model.requests[0].model == JUDGE_MODEL
    [tool] = model.requests[0].tools
    assert tool.input_schema["properties"]["grade"]["enum"] == ["supports", "partly", "does_not_support"]


@pytest.mark.parametrize(
    "failing,grade", [(None, "acceptable"), ("specific", "needs_revision"), ("invents_nothing", "needs_revision")]
)
def test_request_grade_is_worked_out_from_the_four_checks(built, failing, grade):
    items, cases = built
    item = first(items, ItemKind.REQUEST)
    checks = {name: name != failing for name in REQUEST_CHECKS}
    model = ScriptedClient([reply("grade_request", **checks, reasoning="Checked.")])
    judgement = judge_item(model, item, cases[item.case_id])
    assert judgement.grade.grade == grade and judgement.grade.checks == checks
    assert request_grade(checks) == grade


def test_a_bad_grade_is_sent_back_and_can_be_corrected(built):
    items, cases = built
    item = first(items, ItemKind.EVIDENCE)
    model = ScriptedClient(
        [
            reply("grade_evidence", grade="maybe", reasoning="Unsure."),
            reply("grade_evidence", grade="supports", reasoning="Shown."),
        ]
    )
    judgement = judge_item(model, item, cases[item.case_id])
    error = model.requests[1].turns[-1].tool_results[0]
    assert error.is_error and "grade" in error.content
    assert judgement.grade.grade == "supports"


def test_the_wrong_tool_or_a_second_grade_is_refused(built):
    items, _ = built
    tool = GradeTool(first(items, ItemKind.EVIDENCE))
    wrong = tool.run(ToolCall(id="t", name="grade_request", input={}))
    assert wrong.is_error and "use grade_evidence" in wrong.content
    tool.run(ToolCall(id="t", name="grade_evidence", input={"grade": "supports", "reasoning": "Shown."}))
    again = tool.run(ToolCall(id="t", name="grade_evidence", input={"grade": "partly", "reasoning": "Hm."}))
    assert again.is_error and tool.grade.grade == "supports"


def test_a_judge_that_never_grades_is_an_error_not_a_grade(built):
    items, cases = built
    item = first(items, ItemKind.REQUEST)
    judgement = judge_item(ScriptedClient([ModelResponse(stop_reason=StopReason.OTHER)]), item, cases[item.case_id])
    assert judgement.grade is None and judgement.error == "no grade (model_stopped)"


def test_record_then_replay_in_parallel_keeps_item_order(built, tmp_path):
    items, cases = built
    chosen = items[:6] + [i for i in items if i.kind is ItemKind.REQUEST][:2]
    recorded = run_judge(Mode.RECORD, chosen, cases, client=RuleJudge(), recordings=tmp_path, workers=4)
    assert [j.item_id for j in recorded] == [i.item_id for i in chosen]
    assert all(judge_path(tmp_path, i.item_id).exists() for i in chosen)
    replayed = run_judge(Mode.REPLAY, chosen, cases, recordings=tmp_path, workers=4)
    assert [j.grade for j in replayed] == [j.grade for j in recorded]
    assert replayed[-1].grade.grade == "needs_revision"


def test_a_changed_prompt_makes_judge_recordings_stale(built, tmp_path, monkeypatch):
    items, cases = built
    run_judge(Mode.RECORD, items[:1], cases, client=RuleJudge(), recordings=tmp_path)
    edited = tmp_path / "judge_v2.md"
    edited.write_text("A different judge prompt.")
    monkeypatch.setattr("evals.judge.PROMPT", edited)
    [judgement] = run_judge(Mode.REPLAY, items[:1], cases, recordings=tmp_path)
    assert judgement.error == "model call 1 differs from the recording"


def test_missing_recordings_are_errors(built, tmp_path):
    items, cases = built
    [judgement] = run_judge(Mode.REPLAY, items[:1], cases, recordings=tmp_path)
    assert judgement.error.startswith("no recording at")


def test_cost_at_opus_prices(built, tmp_path):
    items, cases = built
    judgements = run_judge(Mode.RECORD, items[:2], cases, client=RuleJudge(), recordings=tmp_path)
    assert cost_usd(judgements) == pytest.approx(2 * (1300 * 4 + 500 * 20) / 1_000_000)


def test_command_line_measures_a_few_items_first(built, tmp_path, capsys):
    items, cases = built
    recordings = tmp_path / "rec"
    # the agent's recordings, which the items are built from, without the committed judge recordings
    shutil.copytree("evals/recordings", recordings, ignore=shutil.ignore_patterns("judge"))
    chosen = judged_items(items, load_sample())[:2]
    run_judge(Mode.RECORD, chosen, cases, client=RuleJudge(), recordings=recordings)
    assert main(["replay", "--limit", "2", "--recordings", str(recordings)]) == 0
    assert "2 items, 0 errors, cost $0.03" in capsys.readouterr().out
    assert main(["replay", "--limit", "3", "--recordings", str(recordings)]) == 1  # the third has no recording


def test_a_judge_recording_with_unused_replies_is_stale(built, tmp_path):
    items, cases = built
    run_judge(Mode.RECORD, items[:1], cases, client=RuleJudge(), recordings=tmp_path)
    path = judge_path(tmp_path, items[0].item_id)
    data = json.loads(path.read_text())
    data["exchanges"].append(data["exchanges"][-1])
    path.write_text(json.dumps(data))
    [judgement] = run_judge(Mode.REPLAY, items[:1], cases, recordings=tmp_path)
    assert judgement.error == "the run made 1 of the 2 recorded model calls"
