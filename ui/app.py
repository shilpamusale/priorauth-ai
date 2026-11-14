"""PriorAuth-AI demo: recorded runs of the matcher, replayed through the real pipeline.

    python -m streamlit run ui/app.py

Text written by the model (reasoning, quotes) is shown with st.text and notes with escaped HTML, so
nothing in a chart or a model reply is ever rendered as markdown or markup.
"""

import json
from pathlib import Path

import streamlit as st

from evals.gold import load_split
from evals.scoring import cost_usd
from priorauth.policies import load_policies
from priorauth.schemas import CriterionStatus, Decision
from ui.data import SPLIT, folders, highlight, recorded_trials, replay

REPO = "https://github.com/shilpamusale/priorauth-ai"
DECISION_BOX = {
    Decision.READY_TO_SUBMIT: st.success,
    Decision.NEEDS_DOCUMENTATION: st.warning,
    Decision.ESCALATE: st.error,
}
STATUS_ICON = {
    CriterionStatus.MET: "✅",
    CriterionStatus.NOT_MET: "❌",
    CriterionStatus.INSUFFICIENT_EVIDENCE: "❔",
}


@st.cache_resource
def load(gold: Path, policy_dir: Path):
    """Policies and gold cases, loaded once per server and checked as they load."""
    policies = load_policies(policy_dir)
    return policies, load_split(gold, SPLIT, policies)


st.set_page_config(page_title="PriorAuth-AI", page_icon="🩺", layout="wide")
st.title("PriorAuth-AI")
st.markdown(
    "Is this prior authorization request **ready to submit** to the payer? An agent reads the patient's chart "
    "with tools and checks each criterion of the coverage policy, quoting the chart as evidence. The decision "
    "is then computed in code, never by the model. Patients are synthetic; the policy is the public Medicare "
    f"coverage document LCD L38773. [Source code]({REPO})"
)

gold, recordings, policy_dir = folders()
policies, cases = load(gold, policy_dir)
trials = recorded_trials(recordings, list(cases))
if not trials:
    st.info("No recorded runs yet. Record one with: `python -m evals.runner record --cases dev_001 --trials 1`")
    st.stop()

pick_case, pick_trial = st.columns([3, 1])
case_id = pick_case.selectbox("Synthetic case", list(trials), format_func=lambda c: f"{c} ({', '.join(cases[c].tags)})")
trial = pick_trial.selectbox("Trial", trials[case_id], format_func=lambda t: f"Trial {t}")
st.caption(
    "Each case was run several times, because the model's answers can vary. Every trial is a recording of a live "
    "run, replayed here through the same code that CI tests. No model is called on this page."
)

case = cases[case_id]
policy = policies[case.policy_id]
result = replay(case, policy, recordings, trial)
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
DECISION_BOX[determination.decision](determination.rationale)
if determination.missing_documentation:
    st.markdown("**Documentation to add before submitting**")
    st.text("\n".join(f"• {gap}" for gap in determination.missing_documentation))

left, right = st.columns([3, 2])
with left:
    st.subheader("Criteria")
    findings = {r.criterion_id: r for r in determination.criteria}
    for criterion in policy.criteria:
        finding = findings.get(criterion.criterion_id)
        expected = case.labels[criterion.criterion_id].status
        status = finding.status if finding else None
        agrees = "matches the gold label" if status is expected else f"gold label: {expected}"
        name = criterion.criterion_id.split(".", 1)[1].replace("_", " ")
        with st.expander(f"{STATUS_ICON.get(status, '⚠️')} {name}: {status or 'no finding'} ({agrees})"):
            st.markdown(f"**Policy:** {criterion.text}")
            if not finding:
                st.markdown("No grounded finding, so this criterion counts as unchecked and the case escalates.")
                continue
            if finding.reasoning:
                st.markdown("**Reasoning**")
                st.text(finding.reasoning)
            if finding.evidence:
                st.markdown("**Evidence**")
                st.text("\n".join(f'{item.note_id}: "{item.quote}"' for item in finding.evidence))

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
                text = answer.content if len(answer.content) <= 400 else answer.content[:400] + " …"
                st.text(("Error: " if answer.is_error else "") + text)

st.caption("Synthetic data for demonstration only. Not for clinical or billing use.")
