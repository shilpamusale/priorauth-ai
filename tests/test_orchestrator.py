"""Multi-agent matching: one evidence agent per criterion, run in parallel, combined in policy order."""

import datetime
import threading
import time

from priorauth.decision import decide
from priorauth.llm.base import ModelResponse, StopReason, ToolCall, Usage
from priorauth.llm.replay import RecordingClient, ReplayClient
from priorauth.llm.scripted import ScriptedClient
from priorauth.matcher import StopCause
from priorauth.orchestrator import run_agents
from priorauth.pipeline import AuditRecord, Pipeline, run_case
from priorauth.schemas import ClinicalNote, Decision, PARequest, Policy, PolicyCriterion

POLICY = Policy(
    policy_id="demo",
    title="Demo policy",
    source="illustrative",
    criteria=[
        PolicyCriterion(criterion_id="demo.duration", text="Pain has lasted at least 3 months."),
        PolicyCriterion(criterion_id="demo.scale", text="A disability scale is recorded."),
        PolicyCriterion(criterion_id="demo.therapy", text="Physical therapy was tried."),
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
QUOTES = {
    "demo.duration": ("n1", "Low back pain since March 2026"),
    "demo.scale": ("n2", "ODI 44% at discharge"),
    "demo.therapy": ("n1", "Started physical therapy"),
}


class EvidenceModel:
    """Answers each agent according to the one criterion its tools allow, so the replies do not depend on
    the order the parallel agents call in. Can be told to stall or stop for one criterion."""

    def __init__(self, slow=(), silent=()):
        self.requests = []
        self._lock = threading.Lock()
        self._slow, self._silent = set(slow), set(silent)

    def complete(self, request):
        with self._lock:
            self.requests.append(request)
        tool = next(t for t in request.tools if t.name == "submit_finding")
        [criterion_id] = tool.input_schema["properties"]["criterion_id"]["enum"]
        if criterion_id in self._slow:
            time.sleep(0.2)  # finishes last, though it comes first in the policy
        if criterion_id in self._silent:
            return ModelResponse(stop_reason=StopReason.OTHER, usage=Usage(input_tokens=100, output_tokens=0))
        note_id, quote = QUOTES[criterion_id]
        finding = {
            "criterion_id": criterion_id,
            "status": "met",
            "evidence": [{"note_id": note_id, "quote": quote}],
            "reasoning": "Documented.",
        }
        call = ToolCall(id=f"toolu_{criterion_id}", name="submit_finding", input=finding)
        return ModelResponse(
            tool_calls=[call], stop_reason=StopReason.TOOL_USE, usage=Usage(input_tokens=1000, output_tokens=100)
        )


def test_each_agent_sees_and_may_submit_only_its_own_criterion():
    model = EvidenceModel()
    run_agents(model, POLICY, REQUEST)
    assert len(model.requests) == 3
    for request in model.requests:
        tool = next(t for t in request.tools if t.name == "submit_finding")
        [criterion_id] = tool.input_schema["properties"]["criterion_id"]["enum"]
        others = [c.criterion_id for c in POLICY.criteria if c.criterion_id != criterion_id]
        assert criterion_id in request.turns[0].text
        assert not any(other in request.turns[0].text for other in others)


def test_findings_come_back_in_policy_order_whatever_order_agents_finish():
    run = run_agents(EvidenceModel(slow={"demo.duration"}), POLICY, REQUEST)
    assert [f.criterion_id for f in run.findings] == ["demo.duration", "demo.scale", "demo.therapy"]
    assert run.turns[1].tool_calls[0].input["criterion_id"] == "demo.duration"


def test_agents_run_in_parallel():
    started = time.perf_counter()
    run_agents(EvidenceModel(slow={"demo.duration", "demo.scale", "demo.therapy"}), POLICY, REQUEST)
    assert time.perf_counter() - started < 0.5  # three 0.2 s agents, not 0.6 s one after another


def test_steps_usage_and_turns_add_up():
    run = run_agents(EvidenceModel(), POLICY, REQUEST)
    assert (run.stop, run.steps, run.skipped) == (StopCause.COMPLETE, 3, [])
    assert run.usage == Usage(input_tokens=3000, output_tokens=300)
    assert len(run.turns) == 9  # per agent: the case, the reply, the tool result


def test_an_agent_that_stops_leaves_its_criterion_unchecked():
    run = run_agents(EvidenceModel(silent={"demo.scale"}), POLICY, REQUEST)
    assert run.skipped == ["demo.scale"] and run.stop is StopCause.MODEL_STOPPED
    assert decide(POLICY, run.findings) is Decision.ESCALATE


def test_the_multi_agent_pipeline_shares_verify_and_decide():
    record = run_case(EvidenceModel(), POLICY, REQUEST, pipeline=Pipeline.MULTI_AGENT)
    assert record.pipeline is Pipeline.MULTI_AGENT
    assert record.determination.decision is Decision.READY_TO_SUBMIT
    assert record.rejected_quotes == []
    assert AuditRecord.model_validate_json(record.model_dump_json()) == record


def test_a_parallel_run_records_and_replays_exactly():
    live = RecordingClient(EvidenceModel(slow={"demo.duration"}))
    recorded = run_case(live, POLICY, REQUEST, pipeline=Pipeline.MULTI_AGENT)
    replay = ReplayClient(live.recording)
    replayed = run_case(replay, POLICY, REQUEST, pipeline=Pipeline.MULTI_AGENT)
    replay.check_finished()
    assert replayed.determination == recorded.determination
    assert replayed.turns == recorded.turns


def test_the_single_agent_stays_the_default():
    findings = [
        {"criterion_id": c, "status": "met", "evidence": [{"note_id": n, "quote": q}], "reasoning": "Documented."}
        for c, (n, q) in QUOTES.items()
    ]
    calls = [ToolCall(id=f"toolu_{i}", name="submit_finding", input=f) for i, f in enumerate(findings)]
    model = ScriptedClient([ModelResponse(tool_calls=calls, stop_reason=StopReason.TOOL_USE)])
    record = run_case(model, POLICY, REQUEST)
    assert record.pipeline is Pipeline.SINGLE_AGENT and len(model.requests) == 1
