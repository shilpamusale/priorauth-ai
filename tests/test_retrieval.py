"""Retrieval proposes a policy from the order text; code checks the policy and the service code."""

import pytest
from embedding import WordHashEmbedding

from priorauth.retrieval import CHUNK_WORDS, Candidate, Outcome, PolicyIndex, lcd_name, passages, route
from priorauth.schemas import Policy, PolicyCriterion

FACET = "https://www.cms.gov/medicare-coverage-database/view/lcd.aspx?lcdid=38773&ver=32"
VEINS = "https://www.cms.gov/medicare-coverage-database/view/lcd.aspx?lcdid=34082&ver=31"
EYELID = "https://www.cms.gov/medicare-coverage-database/view/lcd.aspx?lcdid=33944&ver=28"
SOURCES = {
    FACET: "Facet joint interventions for chronic low back pain: medial branch blocks and radiofrequency ablation "
    "of the lumbar medial branch nerves, with pain relief measured after each diagnostic block.",
    VEINS: "Treatment of varicose veins of the lower extremity: endovenous ablation of the saphenous vein, "
    "sclerotherapy and phlebectomy after a trial of compression stockings.",
    EYELID: "Blepharoplasty of the upper eyelid for dermatochalasis, when excess eyelid skin obstructs the "
    "superior visual field.",
}
FIRST_BLOCK = Policy(
    policy_id="first_block",
    title="t",
    source=FACET,
    service_codes=["64493", "64494"],
    criteria=[PolicyCriterion(criterion_id="first_block.a", text="A")],
)


class FakeIndex:
    """Returns fixed candidates, so each route outcome can be set up exactly."""

    def __init__(self, *candidates: tuple[str, float]) -> None:
        self.candidates = [Candidate(source=s, distance=d) for s, d in candidates]
        self.searched: list[str] = []

    def search(self, text: str) -> list[Candidate]:
        self.searched.append(text)
        return self.candidates


@pytest.fixture(scope="module")
def index():
    return PolicyIndex(SOURCES, WordHashEmbedding())


def test_passages_overlap_so_a_phrase_at_an_edge_is_whole_in_the_next():
    words = [f"w{i}" for i in range(100)]
    chunks = passages(" ".join(words), size=40, overlap=10)
    assert [c.split()[0] for c in chunks] == ["w0", "w30", "w60"]
    assert all(len(c.split()) <= 40 for c in chunks) and chunks[-1].split()[-1] == "w99"
    assert passages("short text") == ["short text"]


def test_lcd_name_reads_the_mcd_url():
    assert lcd_name(FACET) == "LCD L38773 v32"
    assert lcd_name("illustrative") == "illustrative"


def test_search_ranks_each_lcd_once_by_its_closest_passage(index):
    found = index.search("endovenous ablation of the great saphenous vein")
    assert found[0].source == VEINS
    assert len({c.source for c in found}) == len(found)
    assert [c.distance for c in found] == sorted(c.distance for c in found)


def test_an_lcd_is_as_close_as_its_closest_passage():
    near = " ".join(("saphenous vein ablation after compression stockings".split() * 8)[:CHUNK_WORDS])  # 1st passage
    far = " ".join(["eyelid skin visual field"] * 20)  # the LCD's later passages
    found = PolicyIndex({VEINS: f"{near} {far}"}, WordHashEmbedding()).search(near)
    assert found[0].distance < 0.001


def test_indexes_in_one_process_do_not_share_passages():
    only_eyelid = PolicyIndex({EYELID: SOURCES[EYELID]}, WordHashEmbedding())
    assert {c.source for c in only_eyelid.search("medial branch block of the lumbar facet joint")} == {EYELID}


def test_an_index_needs_text():
    with pytest.raises(ValueError, match="no LCD text"):
        PolicyIndex({}, WordHashEmbedding())


def test_matched_when_the_closest_lcd_has_a_policy_covering_the_code(index):
    found = route(index, [FIRST_BLOCK], "Lumbar medial branch blocks, L4-L5 and L5-S1", "64493")
    assert (found.outcome, found.policy_id) == (Outcome.MATCHED, "first_block")
    assert found.candidates[0].source == FACET
    assert "LCD L38773 v32" in found.reason and "64493" in found.reason


def test_no_order_means_no_search():
    fake = FakeIndex((FACET, 0.1))
    for order in (None, ""):
        assert route(fake, [FIRST_BLOCK], order, "64493").outcome is Outcome.NO_ORDER
    assert fake.searched == []


def test_no_match_when_the_closest_lcd_is_beyond_the_limit():
    found = route(FakeIndex((FACET, 0.71), (VEINS, 0.8)), [FIRST_BLOCK], "Total knee arthroplasty", "27447", 0.7)
    assert (found.outcome, found.policy_id) == (Outcome.NO_MATCH, None)
    assert "0.71" in found.reason and "0.7" in found.reason and len(found.candidates) == 2


def test_the_limit_itself_still_matches():
    assert route(FakeIndex((FACET, 0.7)), [FIRST_BLOCK], "MBB", "64493", 0.7).outcome is Outcome.MATCHED


def test_unsupported_when_no_policy_is_built_on_the_closest_lcd():
    found = route(FakeIndex((VEINS, 0.2), (FACET, 0.3)), [FIRST_BLOCK], "Saphenous vein ablation", "36475")
    assert (found.outcome, found.policy_id) == (Outcome.UNSUPPORTED, None)
    assert "LCD L34082 v31" in found.reason


def test_code_not_covered_names_the_codes_the_policies_do_cover():
    found = route(FakeIndex((FACET, 0.2)), [FIRST_BLOCK], "Lumbar medial branch blocks", "15823")
    assert (found.outcome, found.policy_id) == (Outcome.CODE_NOT_COVERED, None)
    assert "15823" in found.reason and "64493, 64494" in found.reason


def test_only_the_closest_lcd_is_considered():
    # the facet policy covers the code, but the order is closest to the vein LCD: a person must look
    found = route(FakeIndex((VEINS, 0.2), (FACET, 0.21)), [FIRST_BLOCK], "Vein ablation", "64493")
    assert found.outcome is Outcome.UNSUPPORTED
