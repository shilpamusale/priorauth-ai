"""PriorAuth-AI demo: recorded runs of the matcher, replayed through the real pipeline.

    python -m streamlit run ui/app.py

Text written by the model (reasoning, quotes) is shown with st.text and notes with escaped HTML, so
nothing in a chart or a model reply is ever rendered as markdown or markup.
"""

import json
from pathlib import Path

import streamlit as st
from pydantic import TypeAdapter, ValidationError

from evals.gaps import TRIAL as DOCUMENTED_TRIAL
from evals.gaps import run_gaps
from evals.gold import load_split
from evals.judge import JUDGE_MODEL
from evals.judge_items import ItemKind
from evals.runner import Mode
from evals.scoring import cost_usd, summarize
from priorauth.pipeline import Pipeline
from priorauth.policies import load_policies, load_sources
from priorauth.retrieval import MAX_DISTANCE, lcd_name, route
from priorauth.schemas import CriterionStatus, Decision, ServiceCode
from ui.data import (
    SPLIT,
    brief,
    build_index,
    calibration,
    dev_routes,
    folders,
    highlight,
    judge_grades,
    judge_note,
    replay_all,
    route_rows,
)

REPO = "https://github.com/shilpamusale/priorauth-ai"
MAX_ORDER_CHARS = 300
DECISION_BOX = {
    Decision.READY_TO_SUBMIT: st.success,
    Decision.NEEDS_DOCUMENTATION: st.warning,
    Decision.ESCALATE: st.error,
}
PIPELINE_LABEL = {
    Pipeline.SINGLE_AGENT: "Single agent (deployed)",
    Pipeline.MULTI_AGENT: "Multi-agent: one agent per criterion",
}
STATUS_ICON = {
    CriterionStatus.MET: "✅",
    CriterionStatus.NOT_MET: "❌",
    CriterionStatus.INSUFFICIENT_EVIDENCE: "❔",
}


@st.cache_resource
def load(gold: Path, policy_dir: Path, recordings: Path):
    """Policies, gold cases, every recorded trial of each pipeline replayed and scored, the documentation agent's
    requests, the judge's grades, and the retrieval index with every dev request routed, once per server."""
    policies = load_policies(policy_dir)
    cases = load_split(gold, SPLIT, policies)
    replays = {pipeline: replay_all(policies, cases, recordings, pipeline) for pipeline in Pipeline}
    requests = {r.case_id: r.run for r in run_gaps(Mode.REPLAY, policies, gold=gold, recordings=recordings) if r.run}
    grades = judge_grades(policies, gold, recordings)
    index = build_index(policy_dir)
    routes = dev_routes(policies, gold, index)
    replayed = {pipeline: results for pipeline, results in replays.items() if results}
    return policies, cases, replayed, requests, grades, index, routes


def is_service_code(code: str) -> bool:
    try:
        TypeAdapter(ServiceCode).validate_python(code)
    except ValidationError:
        return False
    return True


def cost_per_run(summary) -> float | None:
    ran = summary.trials - summary.errors
    return None if summary.cost_usd is None or not ran else summary.cost_usd / ran


st.set_page_config(page_title="PriorAuth-AI", page_icon="🩺", layout="wide")
gold, recordings, policy_dir = folders()
policies, cases, by_pipeline, requests, grades, index, routes = load(gold, policy_dir, recordings)
sources = " · ".join(f"[{p.title}]({p.source})" for p in policies.values())

st.title("PriorAuth-AI")
st.markdown(
    "Is this prior authorization request **ready to submit** to the payer? Retrieval first finds the coverage "
    "policy from the order. An agent then reads the patient's chart with tools and checks each criterion of that "
    "policy, quoting the chart as evidence. The decision is computed in code, never by the model. Patients are "
    f"synthetic; the policies are public Medicare coverage documents: {sources}. [Source code]({REPO})"
)
if not by_pipeline:
    st.info("No recorded runs yet. Record one with: `python -m evals.runner record --cases dev_001 --trials 1`")
    st.stop()

summaries = {pipeline: summarize(list(results.values()), cases) for pipeline, results in by_pipeline.items()}
pipeline = st.radio(
    "Pipeline", list(by_pipeline), format_func=PIPELINE_LABEL.get, horizontal=True, label_visibility="collapsed"
)
results, summary = by_pipeline[pipeline], summaries[pipeline]
ran = summary.trials - summary.errors
cost = cost_per_run(summary)
runs, gates, spend = st.columns(3)
runs.metric("Runs with the expected decision", f"{round(summary.decision_accuracy * ran)} of {ran}")
gates.metric("Hard-gate failures", len(summary.hard_gate_failures))
spend.metric("Model cost per run", "unknown" if cost is None else f"${cost:.2f}")
note = f"{summary.cases} synthetic dev cases, {summary.trials} recorded runs of the live model."
single, multi = (cost_per_run(summaries[p]) if p in summaries else None for p in Pipeline)
if single and multi:
    accuracy = {p: f"{summaries[p].decision_accuracy:.0%}" for p in Pipeline}
    note += (
        f" Decision accuracy: single agent {accuracy[Pipeline.SINGLE_AGENT]}, multi-agent "
        f"{accuracy[Pipeline.MULTI_AGENT]} at {multi / single:.1f} times the cost per run. "
        "The single agent is the one deployed."
    )
st.caption(note + " Held-out cases stay unseen until the final evaluation.")
if summary.errors:
    st.warning(f"{summary.errors} recordings no longer match the code and are not counted.")

with st.expander("How a request finds its policy: retrieval, then a code check, with no model"):
    unsafe = sum(r.unsafe for r in routes)
    st.caption(
        f"ChromaDB ranks passages of {len(load_sources(policy_dir))} Medicare LCDs by how close they are to the "
        f"order. The closest LCD must be within distance {MAX_DISTANCE}, a policy must be built on it, and that "
        "policy must cover the requested code. Otherwise the request goes to a person, before any model call. "
        f"Dev requests: {sum(r.correct for r in routes)} of {len(routes)} routed right; {unsafe} reached a policy "
        "they should not."
    )
    st.dataframe(route_rows(routes), hide_index=True, width="stretch")
    with st.form("try_retrieval"):
        st.markdown("**Try it.** Retrieval runs here on this server; no model is called.")
        # Filled with an example, not a grey placeholder, so pressing the button shows a route at once.
        order = st.text_input("Order, in the clinician's words", value="Bilateral upper eyelid blepharoplasty")
        code = st.text_input("Service code (CPT or HCPCS)", value="15823")
        submitted = st.form_submit_button("Find the policy")
    if submitted:
        order, code = order.strip()[:MAX_ORDER_CHARS], code.strip().upper()
        if not order:
            st.error("Type the order first: the procedure as the clinician wrote it.")
        elif not is_service_code(code):
            st.error("A service code has five characters: a CPT code such as 15823, or a HCPCS code such as J3490.")
        else:
            found = route(index, policies.values(), order, code)
            st.text(f"{found.outcome.value}{': ' + found.policy_id if found.policy_id else ''}\n{found.reason}")
            if found.candidates:
                closest = [{"LCD": lcd_name(c.source), "distance": c.distance} for c in found.candidates]
                st.dataframe(closest, hide_index=True)
trials = {}
for case_id, trial in results:
    trials.setdefault(case_id, []).append(trial)

pick_case, pick_trial = st.columns([3, 1])
case_id = pick_case.selectbox("Synthetic case", list(trials), format_func=lambda c: f"{c} ({', '.join(cases[c].tags)})")
trial = pick_trial.selectbox("Trial", trials[case_id], format_func=lambda t: f"Trial {t}")
st.caption(
    "A case can have several trials, because the model's answers can vary. Every trial is a recording of a live "
    "run, replayed here through the same code that CI tests. No model is called on this page."
)

case = cases[case_id]
policy = policies[case.policy_id]
result = results[(case_id, trial)]
if result.error:
    st.error(f"This recording no longer matches the code: {result.error}")
    st.stop()
record = result.record
determination = record.determination
request = case.request

st.caption(
    f"Service {request.service_code} · Diagnoses {', '.join(request.icd10_codes)} · "
    f"Request date {request.request_date} · {request.payer} · {len(request.notes)} notes in the chart"
)
routed = {r.case.case_id: r for r in routes}.get(case_id)
if routed:
    st.text(f"Order: {request.ordered_service}\nRetrieval: {routed.route.reason}")
    if not routed.correct:
        st.warning(
            "Retrieval would send this request to a person before any model call. The run below names the gold "
            "case's policy directly, so the agent's work on it can still be reviewed."
        )
DECISION_BOX[determination.decision](determination.rationale)
# The documentation agent and the LLM judge both worked from the single agent's first trial.
first_trial = pipeline is Pipeline.SINGLE_AGENT and trial == DOCUMENTED_TRIAL
graded = {(kind, cid): g for (kind, c, cid), g in grades.items() if c == case_id} if first_trial else {}
documented = requests.get(case_id) if first_trial else None
if documented and documented.requests:
    st.markdown("**Requests to the clinic before submitting**")
    st.caption("Written by the documentation agent from this trial's gaps and the chart, and graded by the LLM judge.")
    for written in documented.requests:
        name = written.criterion_id.split(".", 1)[-1].replace("_", " ")
        lines = [f"{name} · {written.note_type}", written.request]
        if grade := graded.get((ItemKind.REQUEST, written.criterion_id)):
            failed = ", ".join(check.replace("_", " ") for check, passed in grade.checks.items() if not passed)
            lines.append(f"Judge: {grade.grade.replace('_', ' ')}" + (f" (fails: {failed})" if failed else ""))
        st.text("\n".join(lines))
elif determination.missing_documentation:
    st.markdown("**Documentation to add before submitting**")
    st.text("\n".join(f"• {gap}" for gap in determination.missing_documentation))

left, right = st.columns([3, 2])
with left:
    st.subheader("Criteria")
    if any(kind is ItemKind.EVIDENCE for kind, _ in graded):
        st.caption(judge_note(calibration(), JUDGE_MODEL) + f" [How it was calibrated]({REPO}#evaluation)")
    elif grades and not first_trial:
        st.caption("Pick trial 1 of the single agent to see the LLM judge's grade for each met criterion.")
    findings = {r.criterion_id: r for r in determination.criteria}
    for criterion in policy.criteria:
        finding = findings.get(criterion.criterion_id)
        expected = case.labels[criterion.criterion_id].status
        status = finding.status if finding else None
        agrees = "matches the gold label" if status is expected else f"gold label: {expected}"
        name = criterion.criterion_id.split(".", 1)[1].replace("_", " ")
        label = f"{STATUS_ICON.get(status, '⚠️')} {name}: {status or 'no finding'} ({agrees})"
        if grade := graded.get((ItemKind.EVIDENCE, criterion.criterion_id)):
            label += f" · judge: {grade.grade.replace('_', ' ')}"
        with st.expander(label):
            st.markdown(f"**Policy:** {criterion.text}")
            if criterion.source_quotes:
                st.caption("From the LCD")
                st.text("\n".join(f'"{quote}"' for quote in criterion.source_quotes))
            if not finding:
                st.markdown("No grounded finding, so this criterion counts as unchecked and the case escalates.")
                continue
            if finding.reasoning:
                st.markdown("**Reasoning**")
                st.text(finding.reasoning)
            if finding.evidence:
                st.markdown("**Evidence**")
                st.text("\n".join(f'{item.note_id}: "{item.quote}"' for item in finding.evidence))
            if grade:
                st.markdown(f"**LLM judge: {grade.grade.replace('_', ' ')}**")
                st.text(grade.reasoning)

with right:
    st.subheader("Against the gold label")
    agreement = "matches" if determination.decision is case.expected_decision else "differs from"
    st.markdown(f"Expected **{case.expected_decision}**; this trial {agreement} it.")
    rows = [
        {
            "check": c.name,
            "passed": "yes" if c.passed else "no",
            "hard gate": "yes" if c.hard_gate else "",
            "detail": c.detail,
        }
        for c in result.checks
    ]
    st.dataframe(rows, hide_index=True, width="stretch")
    st.subheader("Audit record")
    cost = cost_usd(record)
    st.markdown(
        f"- Model `{record.model}`, prompt `{record.prompt_version}`\n"
        f"- Policy fingerprint `{record.policy_fingerprint}`\n"
        f"- Prompt fingerprint `{record.prompt_fingerprint}`\n"
        f"- {record.steps} model calls, stopped: {record.stop}\n"
        f"- Tokens: {record.usage.input_tokens:,} in, {record.usage.output_tokens:,} out"
        + (f", about ${cost:.3f}" if cost is not None else "")
    )
    if record.rejected_quotes:
        st.markdown("**Quotes rejected by verification**")
        st.text("\n".join(record.rejected_quotes))

st.subheader("The chart")
quotes: dict[str, list[str]] = {}
for finding in determination.criteria:
    for item in finding.evidence:
        quotes.setdefault(item.note_id, []).append(item.quote)
for note in request.notes:
    with st.expander(f"{note.note_id} · {note.date} · {note.note_type}", expanded=note.note_id in quotes):
        st.html(f"<div style='line-height:1.6'>{highlight(note.text, quotes.get(note.note_id, []))}</div>")

with st.expander("Every tool call in this run"):
    answers = {answer.tool_call_id: answer for turn in record.turns for answer in turn.tool_results}
    number = 0
    for turn in record.turns:
        if turn.role == "assistant" and turn.text:
            st.text(f"Model: {turn.text}")
        for call in turn.tool_calls:
            number += 1
            st.markdown(f"**{number}. {call.name}**")
            st.code(json.dumps(call.input), language="json", wrap_lines=True)
            if answer := answers.get(call.id):
                st.text(brief(answer))

st.caption("Synthetic data for demonstration only. Not for clinical or billing use.")
