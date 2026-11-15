"""Judge items and the hand-labelled calibration sample, built from the committed recordings."""

from collections import Counter
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from evals.judge_items import (
    GRADES,
    HumanLabel,
    ItemKind,
    build_items,
    check_label,
    choose_sample,
    load_labels,
    load_sample,
    save_labels,
)
from priorauth.policies import load_policies

LABEL_PAGE = Path(__file__).resolve().parents[1] / "ui" / "label.py"


@pytest.fixture(scope="module")
def built():
    return build_items(load_policies(Path("policies")))


def test_every_met_finding_and_request_becomes_an_item(built):
    items, _ = built
    kinds = Counter(item.kind for item in items)
    assert (kinds[ItemKind.EVIDENCE], kinds[ItemKind.REQUEST]) == (99, 8)
    assert len({item.item_id for item in items}) == len(items)


def test_item_ids_do_not_reveal_the_kind(built):
    items, _ = built
    assert all(item.item_id.startswith("item_") and len(item.item_id) == 15 for item in items)
    assert not any(kind in item.item_id for item in items for kind in ItemKind)


def test_a_planted_item_offers_quotes_from_the_same_case_for_another_criterion(built):
    items, _ = built
    evidence = {(i.case_id, i.criterion_id): i.quotes for i in items if i.kind is ItemKind.EVIDENCE}
    for planted in (i for i in items if i.kind is ItemKind.PLANTED):
        assert planted.quotes in [q for (case, _), q in evidence.items() if case == planted.case_id]
        assert planted.quotes != evidence.get((planted.case_id, planted.criterion_id))


def test_the_committed_sample_is_the_one_the_code_chooses(built):
    items, _ = built
    assert load_sample() == choose_sample(items)


def test_the_sample_mixes_the_kinds_and_never_shows_one_criterion_twice(built):
    items, _ = built
    by_id = {item.item_id: item for item in items}
    sample = [by_id[item_id] for item_id in load_sample()]
    assert Counter(item.kind for item in sample) == {ItemKind.EVIDENCE: 24, ItemKind.PLANTED: 8, ItemKind.REQUEST: 8}
    pairs = [(i.case_id, i.criterion_id) for i in sample if i.kind is not ItemKind.REQUEST]
    assert len(pairs) == len(set(pairs))
    assert [i.kind for i in sample[:8]] != [ItemKind.PLANTED] * 8  # shuffled, not grouped


def test_labels_round_trip_in_sample_order(tmp_path):
    path = tmp_path / "labels.yaml"
    sample = ["item_b", "item_a"]
    save_labels(
        {"item_a": HumanLabel(label="supports"), "item_b": HumanLabel(label="partly", note="weak")}, sample, path
    )
    assert list(load_labels(path)) == ["item_b", "item_a"]
    assert load_labels(path)["item_b"].note == "weak"
    assert load_labels(tmp_path / "missing.yaml") == {}


def test_a_label_must_suit_the_kind_of_item(built):
    items, _ = built
    request = next(i for i in items if i.kind is ItemKind.REQUEST)
    check_label(request, HumanLabel(label="acceptable"))
    with pytest.raises(ValueError, match="not one of"):
        check_label(request, HumanLabel(label="supports"))
    assert GRADES[ItemKind.PLANTED] == GRADES[ItemKind.EVIDENCE]  # same choices, so planted items look the same


def label_page(monkeypatch, labels):
    monkeypatch.setenv("PRIORAUTH_LABELS", str(labels))
    app = AppTest.from_file(str(LABEL_PAGE), default_timeout=60)
    app.run()
    assert not app.exception, app.exception
    return app


def test_the_label_page_saves_a_label_and_moves_on(monkeypatch, tmp_path, built):
    items, _ = built
    labels = tmp_path / "labels.yaml"
    app = label_page(monkeypatch, labels)
    assert app.caption[0].value.startswith("Item 1 of 40")
    page_text = " ".join(str(element.value) for element in [*app.markdown, *app.text, *app.caption])
    assert "planted" not in page_text.lower()
    first = load_sample()[0]
    grade = GRADES[next(i.kind for i in items if i.item_id == first)][0]
    app.radio[0].set_value(grade)  # inside a form, a choice is sent with the submit button
    app.button[0].click().run()
    assert first in load_labels(labels)
    assert app.caption[0].value.startswith("Item 2 of 40")


def test_the_label_page_says_when_everything_is_labelled(monkeypatch, tmp_path, built):
    items, _ = built
    by_id = {item.item_id: item for item in items}
    sample = load_sample()
    labels = tmp_path / "labels.yaml"
    save_labels({i: HumanLabel(label=GRADES[by_id[i].kind][0]) for i in sample}, sample, labels)
    app = label_page(monkeypatch, labels)
    assert app.success[0].value.startswith("All labelled.")


def test_a_planted_item_looks_exactly_like_a_real_one(monkeypatch, tmp_path, built):
    items, _ = built
    by_id = {item.item_id: item for item in items}
    sample = load_sample()
    planted = next(n for n, i in enumerate(sample, 1) if by_id[i].kind is ItemKind.PLANTED)
    real = next(n for n, i in enumerate(sample, 1) if by_id[i].kind is ItemKind.EVIDENCE)

    def headings(number):
        app = label_page(monkeypatch, tmp_path / "labels.yaml")
        app.number_input[0].set_value(number).run()
        shown = [m.value for m in app.markdown if m.value.startswith("**") and " · " not in m.value]
        return shown, " ".join(str(e.value) for e in [*app.markdown, *app.text, *app.caption]).lower()

    planted_headings, planted_text = headings(planted)
    assert planted_headings == headings(real)[0]
    assert "planted" not in planted_text and "evidence" not in planted_text
