"""Label judge items by hand, blind, to calibrate the LLM judge. Run it locally; it is never deployed.

    python -m streamlit run ui/label.py

One sampled item at a time: the criterion, the passages offered as evidence (or the documentation request)
and the chart. It never shows the kind of item or any judge grade. Each label is saved to
evals/judge/labels.yaml as soon as it is given.
"""

import os
from pathlib import Path

import streamlit as st

from evals.judge_items import GRADES, LABELS, HumanLabel, ItemKind, build_items, load_labels, load_sample, save_labels
from priorauth.policies import load_policies
from ui.data import folders, highlight

QUESTION = {
    "evidence": "Do these passages, on their own, show that the criterion is met?",
    "request": "Could this request go to the clinic as written?",
}
GRADE_TEXT = {
    "supports": "Supports: the passages show the criterion is met",
    "partly": "Partly: relevant, but not enough on their own",
    "does_not_support": "Does not support: they do not show this criterion",
    "acceptable": "Acceptable: specific to this chart, asks only for what the criterion needs, invents nothing",
    "needs_revision": "Needs revision: vague, asks for too much, invents something, or unclear where to record it",
}


@st.cache_resource
def load(gold: Path, policy_dir: Path, recordings: Path):
    items, cases = build_items(load_policies(policy_dir), gold, recordings)
    return {item.item_id: item for item in items}, cases


st.set_page_config(page_title="Label judge items", layout="wide")
gold, recordings, policy_dir = folders()
labels_path = Path(os.environ.get("PRIORAUTH_LABELS", LABELS))
items, cases = load(gold, policy_dir, recordings)
sample = load_sample(Path(os.environ.get("PRIORAUTH_SAMPLE", "evals/judge/sample.yaml")))
labels = load_labels(labels_path)

st.title("Label judge items")
done = sum(1 for item in sample if item in labels)
st.progress(done / len(sample), text=f"{done} of {len(sample)} labelled")
if done == len(sample):
    st.success(f"All labelled. Commit {labels_path} on its own branch; the judge runs only after that.")

first_open = next((n for n, item in enumerate(sample) if item not in labels), 0)
number = st.number_input("Item", min_value=1, max_value=len(sample), value=first_open + 1, step=1)
item = items[sample[number - 1]]
case = cases[item.case_id]
st.caption(f"Item {number} of {len(sample)} · case {item.case_id} · request date {case.request.request_date}")

st.markdown("**Criterion**")
st.text(item.criterion_text)
if item.kind is ItemKind.REQUEST:
    st.markdown(f"**The chart does not document this criterion. Request to the clinic** (to go in: {item.note_type})")
    st.text(item.request)
    shown_notes, quotes = case.request.notes, {}
    question = QUESTION["request"]
else:
    st.markdown("**The agent marked this criterion met, citing:**")
    st.text("\n".join(f'{q.note_id}: "{q.quote}"' for q in item.quotes))
    cited = {q.note_id for q in item.quotes}
    shown_notes = [note for note in case.request.notes if note.note_id in cited]
    quotes = {note_id: [q.quote for q in item.quotes if q.note_id == note_id] for note_id in cited}
    question = QUESTION["evidence"]
for note in shown_notes:
    st.markdown(f"**{note.note_id} · {note.date} · {note.note_type}**")
    st.html(f"<div style='line-height:1.6'>{highlight(note.text, quotes.get(note.note_id, []))}</div>")
with st.expander("Whole chart"):
    for note in case.request.notes:
        st.markdown(f"**{note.note_id} · {note.date} · {note.note_type}**")
        st.text(note.text)

current = labels.get(item.item_id)
options = GRADES[item.kind]
with st.form(f"label_{item.item_id}"):
    grade = st.radio(
        question, options, format_func=GRADE_TEXT.get, index=options.index(current.label) if current else None
    )
    note = st.text_input("Note (optional)", value=current.note if current else "")
    if st.form_submit_button("Save and go to the next item") and grade:
        labels[item.item_id] = HumanLabel(label=grade, note=note)
        save_labels(labels, sample, labels_path)
        st.rerun()
