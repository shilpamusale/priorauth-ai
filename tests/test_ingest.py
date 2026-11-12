"""The extractor keeps the coverage section intact and strips what the browser added when printing."""

import pytest
from fpdf import FPDF

from priorauth.ingest import IngestError, coverage_section, extract, main, normalize, page_body, read_pdf

URL = "https://www.cms.gov/medicare-coverage-database/view/lcd.aspx?lcdid=38773&ver=32"
START = "Coverage Indications, Limitations, and/or Medical Necessity"


def page(body, number, url=URL):
    return f"9/4/26, 6:58 PM LCD - Facet Joint Interventions (L38773)\n{body}\n{url} {number}/2"


PAGES = [
    page(f"Background text\n{START}\n1. Pain present for minimum of 3 months with documented", 1),
    page("failure to respond to noninvasive conservative management\nSummary of Evidence\nStudies", 2),
]


def test_normalize_fixes_ligatures_quotes_hyphens_and_spaces():
    assert normalize("de\ufb01cit") == "deficit"
    assert normalize("the patient\u2019s \u201cpain\u201d") == "the patient's \"pain\""
    assert normalize("long-\nterm") == "long-term"
    assert normalize("non-facet  pathology\n of the") == "non-facet pathology of the"


def test_page_body_strips_print_header_and_footer():
    lines, url = page_body(page("line one\nline two", 4))
    assert lines == ["line one", "line two"]
    assert url == URL


def test_page_without_print_marks_is_unchanged():
    assert page_body("line one\nline two") == (["line one", "line two"], None)


def test_coverage_section_stops_at_the_next_heading():
    lines = ["intro", START, "criteria", "Summary of Evidence", "studies"]
    assert coverage_section(lines) == [START, "criteria"]


@pytest.mark.parametrize(
    "lines",
    [
        ["intro", "Summary of Evidence"],  # no coverage heading
        [START, "criteria", START, "Summary of Evidence"],  # two coverage headings
        [START, "criteria"],  # no end heading
    ],
)
def test_unexpected_layout_is_an_error(lines):
    with pytest.raises(IngestError):
        coverage_section(lines)


def test_extract_joins_pages_without_print_marks():
    text = extract(PAGES)
    assert text.startswith(f"# source: {URL}\n")
    assert "6:58 PM" not in text and "1/2" not in text and "Background" not in text and "Studies" not in text
    sentence = "Pain present for minimum of 3 months with documented failure to respond"
    assert sentence in normalize(text)  # the sentence crosses the page break


def test_extract_stores_plain_letters_for_ligatures():
    text = extract([page(f"{START}\nfunctional de\ufb01cit\nSummary of Evidence", 1)])
    assert "functional deficit" in text and "\ufb01" not in text


def test_extract_needs_one_source_url():
    with pytest.raises(IngestError, match="found 2"):
        extract([PAGES[0], page("more", 2, url=URL.replace("ver=32", "ver=31"))])
    with pytest.raises(IngestError, match="found 0"):
        extract([f"{START}\ncriteria\nSummary of Evidence"])


def test_read_pdf_keeps_an_offset_bold_word_in_its_sentence(tmp_path):
    """The real LCD prints words such as "ALL" a few points off the line; they must stay in place."""
    pdf = FPDF(unit="pt")
    pdf.add_page()
    x = 72
    for word, style, shift in [("patients who meet ", "", 0), ("ALL ", "B", 4), ("the following criteria:", "", 0)]:
        pdf.set_font("Helvetica", style=style, size=10)
        pdf.text(x, 100 + shift, word)
        x += pdf.get_string_width(word)
    path = tmp_path / "bold.pdf"
    pdf.output(str(path))
    assert read_pdf(path)[0].splitlines() == ["patients who meet ALL the following criteria:"]


def test_main_reads_a_pdf_and_writes_the_section(tmp_path):
    """End to end on a small generated PDF, so CI exercises pdfplumber without the real LCD."""
    pdf = FPDF()
    pdf.set_font("Helvetica", size=10)
    for page_text in PAGES:
        pdf.add_page()
        for line in page_text.splitlines():
            pdf.cell(0, 6, line, new_x="LMARGIN", new_y="NEXT")
    pdf_path, out = tmp_path / "lcd.pdf", tmp_path / "sources" / "l38773_v32.txt"
    pdf.output(str(pdf_path))
    main([str(pdf_path), str(out)])
    assert out.read_text() == extract(PAGES)