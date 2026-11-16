"""Local reviewer screen: work the review queue, blind to the gold label.

    python -m streamlit run ui/review.py

Each case the agent escalated, or whose trials disagree, is listed with the reasons. The reviewer sees the
chart with the agent's quotes highlighted and every trial's status for each criterion, then confirms or
corrects each criterion with a reason, and a quote for anything marked met. Code decides again, and the
review is saved to evals/reviews/<case_id>.yaml. This page writes files, so it is never deployed, and it
never reads a gold label.
"""

import datetime
import os
from pathlib import Path

import streamlit as st

from evals.gold import load_split
from evals.reviews import REVIEWED_TRIAL, REVIEWS, load_reviews, recorded_trials, review_queue, save_review
from priorauth.policies import load_policies
from priorauth.review import Review, ReviewError, reviewed_determination
from priorauth.schemas import CriterionStatus
from ui.data import KEEP, folders, highlight, overrides_from_answers

CHOICES = [KEEP, *(status.value for status in CriterionStatus)]


@st.cache_resource
def load(gold: Path, policy_dir: Path, recordings: Path):
    policies = load_policies(policy_dir)
    cases = load_split(gold, "dev", policies)
    return policies, cases, review_queue(recorded_trials(policies, gold, recordings))


st.set_page_config(page_title="PriorAuth-AI review", page_icon="🩺", layout="wide")
gold, recordings, policy_dir = folders()
reviews_dir = Path(os.environ.get("PRIORAUTH_REVIEWS", REVIEWS))
policies, cases, queue = load(gold, policy_dir, recordings)
done = load_reviews(reviews_dir)

st.title("Review queue")
st.caption(
    f"{len(queue)} cases need a person: the agent escalated them, or its trials disagree. "
    f"{sum(item.case_id in done for item in queue)} reviewed. Gold labels are not shown."
)
if not queue:
    st.info("Nothing to review.")
    st.stop()

item = st.selectbox(
    "Case",
    queue,
    format_func=lambda i: f"{i.case_id}{' (reviewed)' if i.case_id in done else ''}: {'; '.join(i.reasons)}",
    key="case",
)
case = cases[item.case_id]
policy, request = policies[case.policy_id], case.request
agent = item.trials[REVIEWED_TRIAL]
findings = {r.criterion_id: r for r in agent.criteria}
previous = done.get(item.case_id)

st.markdown("**Why it is here**")
st.text("\n".join(f"- {reason}" for reason in item.reasons))
st.caption(
    f"Order: {request.ordered_service} · service {request.service_code} · diagnoses "
    f"{', '.join(request.icd10_codes)} · request date {request.request_date}"
)
st.text(f"Agent's decision on trial {REVIEWED_TRIAL}: {agent.decision}\n{agent.rationale}")

left, right = st.columns([3, 2])
with left, st.form(f"review:{item.case_id}"):
    st.text_input("Reviewer", value=previous.reviewer if previous else "", key=f"{item.case_id}:reviewer")
    for criterion in policy.criteria:
        cid = criterion.criterion_id
        finding = findings.get(cid)
        st.markdown(f"**{cid.split('.', 1)[1].replace('_', ' ')}**")
        st.caption(criterion.text)
        by_trial = ", ".join(
            f"trial {t}: {next((r.status.value for r in d.criteria if r.criterion_id == cid), 'no finding')}"
            for t, d in sorted(item.trials.items())
        )
        lines = [by_trial]
        if finding:
            lines += [f"Agent: {finding.reasoning}"] if finding.reasoning else []
            lines += [f'{e.note_id}: "{e.quote}"' for e in finding.evidence]
        st.text("\n".join(lines))
        earlier = previous.overrides.get(cid) if previous else None
        status, reason, note, quote = st.columns([2, 3, 1.2, 3])
        status.selectbox(
            "Your status",
            CHOICES,
            index=CHOICES.index(earlier.status.value) if earlier else 0,
            key=f"{item.case_id}:{cid}:status",
        )
        reason.text_input(
            "Reason, if you change it", value=earlier.reason if earlier else "", key=f"{item.case_id}:{cid}:reason"
        )
        note_ids = ["", *(n.note_id for n in request.notes)]
        cited = earlier.evidence[0] if earlier and earlier.evidence else None
        note.selectbox(
            "Note",
            note_ids,
            index=note_ids.index(cited.note_id) if cited else 0,
            format_func=lambda note_id: note_id or "none",
            key=f"{item.case_id}:{cid}:note",
        )
        quote.text_input(
            "Quote, if you mark it met", value=cited.quote if cited else "", key=f"{item.case_id}:{cid}:quote"
        )
    st.text_area("Comment", value=previous.comment if previous else "", key=f"{item.case_id}:comment")
    submitted = st.form_submit_button("Save review")

with right:
    st.subheader("The chart")
    quotes: dict[str, list[str]] = {}
    for finding in agent.criteria:
        for evidence in finding.evidence:
            quotes.setdefault(evidence.note_id, []).append(evidence.quote)
    for note in request.notes:
        with st.expander(f"{note.note_id} · {note.date} · {note.note_type}", expanded=True):
            st.html(f"<div style='line-height:1.6'>{highlight(note.text, quotes.get(note.note_id, []))}</div>")

if submitted:
    reviewer = st.session_state[f"{item.case_id}:reviewer"].strip()
    overrides, problems = overrides_from_answers(item.case_id, policy.criteria, findings, st.session_state)
    if not reviewer:
        problems.insert(0, "Reviewer: give your name")
    if problems:
        st.error("Not saved.\n\n" + "\n".join(f"- {p}" for p in problems))
    else:
        review = Review(
            case_id=item.case_id,
            policy_id=policy.policy_id,
            trial=REVIEWED_TRIAL,
            reviewer=reviewer,
            reviewed_at=datetime.datetime.now(datetime.UTC),
            overrides=overrides,
            comment=st.session_state[f"{item.case_id}:comment"].strip(),
        )
        try:
            determination = reviewed_determination(policy, request, agent.criteria, review)
        except ReviewError as e:
            st.error(f"Not saved. {e}")
        else:
            path = save_review(review, reviews_dir)
            st.success(f"Saved to {path}. With your review, decide() returns {determination.decision}.")
            st.text(determination.rationale)
