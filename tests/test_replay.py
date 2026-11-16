"""Record a run, replay it without a model, and refuse to replay a run that has changed."""

import datetime

import pytest

from priorauth.llm.base import ModelClient, ModelRequest, ModelResponse, StopReason, ToolCall, ToolResult, Turn, Usage
from priorauth.llm.replay import (
    RecordingClient,
    ReplayClient,
    StaleRecording,
    load_recording,
    request_fingerprint,
    save_recording,
)
from priorauth.llm.scripted import ScriptedClient
from priorauth.pipeline import run_case
from priorauth.schemas import ClinicalNote, PARequest, Policy, PolicyCriterion

POLICY = Policy(
    policy_id="demo",
    title="Demo policy",
    source="illustrative",
    service_codes=["64493"],
    criteria=[
        PolicyCriterion(criterion_id="demo.duration", text="Pain has lasted at least 3 months."),
        PolicyCriterion(criterion_id="demo.scale", text="A disability scale is recorded."),
    ],
)
REQUEST = PARequest(
    case_id="dev_900",
    payer="Medicare",
    request_date=datetime.date(2026, 10, 1),
    service_code="64493",
    icd10_codes=["M54.50"],
    notes=[
        ClinicalNote(
            note_id="n1",
            date=datetime.date(2026, 6, 1),
            note_type="primary care visit",
            text="Low back pain since March 2026. Started physical therapy.",
        ),
        ClinicalNote(
            note_id="n2",
            date=datetime.date(2026, 9, 1),
            note_type="physical therapy discharge summary",
            text="ODI 44% at discharge. No change with physical therapy.",
        ),
    ],
)


def model_replies():
    """What a live model might say on REQUEST: read a note, then submit both findings."""
    read = ToolCall(id="toolu_1", name="read_note", input={"note_id": "n2"})
    submits = [
        ToolCall(
            id="toolu_2",
            name="submit_finding",
            input={
                "criterion_id": "demo.duration",
                "status": "met",
                "evidence": [{"note_id": "n1", "quote": "Low back pain since March 2026"}],
                "reasoning": "Pain since March.",
            },
        ),
        ToolCall(
            id="toolu_3",
            name="submit_finding",
            input={
                "criterion_id": "demo.scale",
                "status": "met",
                "evidence": [{"note_id": "n2", "quote": "ODI 44% at discharge"}],
                "reasoning": "ODI recorded.",
            },
        ),
    ]
    usage = Usage(input_tokens=900, output_tokens=120)
    return [
        ModelResponse(tool_calls=[read], stop_reason=StopReason.TOOL_USE, usage=usage),
        ModelResponse(tool_calls=submits, stop_reason=StopReason.TOOL_USE, usage=usage),
    ]


def recorded():
    """Record one live run (the scripted client stands in for Claude) and return it with its audit record."""
    live = RecordingClient(ScriptedClient(model_replies()))
    record = run_case(live, POLICY, REQUEST)
    return live.recording, record


def test_replay_reproduces_the_live_run():
    recording, live_record = recorded()
    replay = ReplayClient(recording)
    replayed = run_case(replay, POLICY, REQUEST)
    replay.check_finished()
    assert replayed.determination == live_record.determination
    assert replayed.turns == live_record.turns
    assert replayed.usage == live_record.usage


def test_recording_keeps_one_exchange_per_model_call():
    recording, _ = recorded()
    assert len(recording.exchanges) == 2
    assert recording.exchanges[1].response == model_replies()[1]


def test_a_recording_survives_saving_and_loading(tmp_path):
    recording, _ = recorded()
    path = tmp_path / "recordings" / "dev_900" / "trial_1.json"
    save_recording(recording, path)
    assert load_recording(path) == recording


def test_changed_chart_makes_the_recording_stale():
    recording, _ = recorded()
    notes = [REQUEST.notes[0], REQUEST.notes[1].model_copy(update={"text": "ODI 46% at discharge."})]
    changed = REQUEST.model_copy(update={"notes": notes})
    with pytest.raises(StaleRecording, match="model call 2 differs"):  # the note read in call 1 now says something else
        run_case(ReplayClient(recording), POLICY, changed)


def test_changed_policy_makes_the_recording_stale():
    recording, _ = recorded()
    criteria = [*POLICY.criteria[:1], POLICY.criteria[1].model_copy(update={"text": "A pain scale is recorded."})]
    with pytest.raises(StaleRecording, match="model call 1 differs"):
        run_case(ReplayClient(recording), POLICY.model_copy(update={"criteria": criteria}), REQUEST)


def test_changed_model_makes_the_recording_stale():
    recording, _ = recorded()
    with pytest.raises(StaleRecording, match="model call 1 differs"):
        run_case(ReplayClient(recording), POLICY, REQUEST, model="claude-haiku-5-5")


def test_a_run_that_needs_more_calls_than_recorded_is_stale():
    recording, _ = recorded()
    short = recording.model_copy(update={"exchanges": recording.exchanges[:1]})
    with pytest.raises(StaleRecording, match="more than the 1 recorded"):
        run_case(ReplayClient(short), POLICY, REQUEST)


def test_a_run_that_stops_early_is_stale():
    recording, _ = recorded()
    longer = recording.model_copy(update={"exchanges": [*recording.exchanges, recording.exchanges[-1]]})
    replay = ReplayClient(longer)
    run_case(replay, POLICY, REQUEST)  # finishes after 2 calls; the recording expects 3
    with pytest.raises(StaleRecording, match="made 2 of the 3"):
        replay.check_finished()


def test_fingerprint_ignores_dict_order_but_not_content():
    def request(tool_input, max_tokens=4096):
        call = ToolCall(id="toolu_1", name="read_note", input=tool_input)
        return ModelRequest(
            model="claude-sonnet-5-5",
            system="s",
            max_tokens=max_tokens,
            turns=[
                Turn(role="user", text="go"),
                Turn(role="assistant", tool_calls=[call]),
                Turn(role="user", tool_results=[ToolResult(tool_call_id="toolu_1", content="note text")]),
            ],
        )

    first = request({"note_id": "n1", "page": 1})
    assert request_fingerprint(first) == request_fingerprint(request({"page": 1, "note_id": "n1"}))
    assert request_fingerprint(first) != request_fingerprint(request({"note_id": "n2", "page": 1}))
    assert request_fingerprint(first) != request_fingerprint(request({"note_id": "n1", "page": 1}, max_tokens=1024))


def test_both_clients_satisfy_the_interface():
    assert isinstance(RecordingClient(ScriptedClient([])), ModelClient)
    assert isinstance(ReplayClient(recorded()[0]), ModelClient)


def test_parallel_callers_get_their_own_replies_in_any_order():
    """Agents running in parallel send requests in no fixed order; each still gets the reply recorded for it."""
    first, second = (
        ModelRequest(model="claude-sonnet-5-5", system="s", turns=[Turn(role="user", text=text)])
        for text in ("criterion A", "criterion B")
    )
    replies = {text: ModelResponse(text=f"reply to {text}", stop_reason=StopReason.END_TURN) for text in ("A", "B")}
    live = RecordingClient(ScriptedClient([replies["A"], replies["B"]]))
    live.complete(first)
    live.complete(second)
    replay = ReplayClient(live.recording)
    assert replay.complete(second).text == "reply to B"  # asked in the opposite order
    assert replay.complete(first).text == "reply to A"
    replay.check_finished()


def test_a_repeated_request_gets_its_replies_in_recorded_order_and_no_more():
    request = ModelRequest(model="claude-sonnet-5-5", system="s", turns=[Turn(role="user", text="same")])
    live = RecordingClient(ScriptedClient([ModelResponse(text=t, stop_reason=StopReason.END_TURN) for t in "12"]))
    live.complete(request)
    live.complete(request)
    replay = ReplayClient(live.recording)
    assert [replay.complete(request).text, replay.complete(request).text] == ["1", "2"]
    with pytest.raises(StaleRecording, match="more than the 2 recorded"):
        replay.complete(request)


def test_both_clients_are_safe_to_share_between_threads():
    from concurrent.futures import ThreadPoolExecutor

    requests = [
        ModelRequest(model="claude-sonnet-5-5", system="s", turns=[Turn(role="user", text=f"criterion {i}")])
        for i in range(50)
    ]

    class Echo:
        def complete(self, request):
            return ModelResponse(text=request.turns[0].text, stop_reason=StopReason.END_TURN)

    live = RecordingClient(Echo())
    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(live.complete, requests))
    assert len(live.recording.exchanges) == 50
    replay = ReplayClient(live.recording)
    with ThreadPoolExecutor(max_workers=8) as pool:
        answers = list(pool.map(replay.complete, requests))
    assert [a.text for a in answers] == [r.turns[0].text for r in requests]
    replay.check_finished()
