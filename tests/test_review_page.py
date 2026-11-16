"""The local reviewer screen, driven with Streamlit's AppTest on the committed recordings."""

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from evals.reviews import load_reviews

PAGE = Path(__file__).resolve().parents[1] / "ui" / "review.py"
DURATION = "dev_005:l38773_first_mbb.duration"


@pytest.fixture
def screen(monkeypatch, tmp_path):
    monkeypatch.setenv("PRIORAUTH_REVIEWS", str(tmp_path))
    app = AppTest.from_file(str(PAGE), default_timeout=60)
    app.run()
    assert not app.exception, app.exception
    return app, tmp_path


def save(app):
    app.button[0].click().run()
    assert not app.exception, app.exception
    return app


def test_the_queue_lists_each_case_with_its_reasons(screen):
    app, _ = screen
    assert app.selectbox(key="case").options[0].startswith("dev_005: the agent escalated")
    assert len(app.selectbox(key="case").options) == 7
    assert "7 cases need a person" in app.caption[0].value and "0 reviewed" in app.caption[0].value


def test_a_review_needs_a_reviewer(screen):
    app, reviews = screen
    save(app)
    assert app.error[0].value.startswith("Not saved.") and "give your name" in app.error[0].value
    assert load_reviews(reviews) == {}


def test_confirming_the_agent_saves_a_review_and_code_decides_again(screen):
    app, reviews = screen
    app.text_input(key="dev_005:reviewer").set_value("Ishi")
    save(app)
    assert "With your review, decide() returns escalate." in app.success[0].value
    saved = load_reviews(reviews)["dev_005"]
    assert (saved.reviewer, saved.trial, saved.overrides) == ("Ishi", 1, {})


def test_a_change_needs_a_reason(screen):
    app, reviews = screen
    app.text_input(key="dev_005:reviewer").set_value("Ishi")
    app.selectbox(key=f"{DURATION}:status").set_value("insufficient_evidence")
    save(app)
    assert "say why you changed it" in app.error[0].value and load_reviews(reviews) == {}


def test_marking_met_needs_a_quote_that_is_in_the_note(screen):
    app, reviews = screen
    app.text_input(key="dev_005:reviewer").set_value("Ishi")
    app.selectbox(key=f"{DURATION}:status").set_value("met")
    app.text_input(key=f"{DURATION}:reason").set_value("Counting from the first visit.")
    save(app)
    assert "needs a quote" in app.error[0].value
    app.selectbox(key=f"{DURATION}:note").set_value("n1")
    app.text_input(key=f"{DURATION}:quote").set_value("Axial low back pain for 6 weeks")
    save(app)
    assert "is not in n1" in app.error[0].value and load_reviews(reviews) == {}
    app.selectbox(key=f"{DURATION}:note").set_value("n4")
    save(app)
    assert "With your review, decide() returns ready_to_submit." in app.success[0].value
    assert load_reviews(reviews)["dev_005"].overrides["l38773_first_mbb.duration"].evidence[0].note_id == "n4"


def test_a_reviewed_case_is_marked_and_its_review_comes_back_into_the_form(screen):
    app, _ = screen
    app.text_input(key="dev_005:reviewer").set_value("Ishi")
    app.selectbox(key=f"{DURATION}:status").set_value("not_met")
    app.text_input(key=f"{DURATION}:reason").set_value("Six weeks is short of three months.")
    save(app)
    again = AppTest.from_file(str(PAGE), default_timeout=60)
    again.run()
    assert again.selectbox(key="case").options[0].startswith("dev_005 (reviewed)")
    assert "1 reviewed" in again.caption[0].value
    assert again.selectbox(key=f"{DURATION}:status").value == "not_met"


def test_the_screen_never_reads_a_gold_label():
    source = PAGE.read_text()
    assert ".labels" not in source and "expected_decision" not in source


def test_a_criterion_the_agent_left_unchecked_needs_a_status():
    from priorauth.schemas import PolicyCriterion
    from ui.data import KEEP, overrides_from_answers

    criteria = [PolicyCriterion(criterion_id="p.a", text="A")]
    answers = {"c:p.a:status": KEEP, "c:p.a:reason": "", "c:p.a:note": "", "c:p.a:quote": ""}
    overrides, problems = overrides_from_answers("c", criteria, {}, answers)
    assert overrides == {} and problems == ["p.a: the agent left this unchecked, so give it a status"]
    checked = answers | {"c:p.a:status": "insufficient_evidence", "c:p.a:reason": "Nothing in the chart."}
    overrides, problems = overrides_from_answers("c", criteria, {}, checked)
    assert problems == [] and overrides["p.a"].status == "insufficient_evidence"
