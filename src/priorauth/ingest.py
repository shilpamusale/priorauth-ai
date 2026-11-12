"""Extract the coverage section from a Local Coverage Determination (LCD) PDF.

The PDF is the LCD page from the CMS Medicare Coverage Database, printed to PDF from a browser. It is
downloaded by hand and kept out of the repo. The extracted text is committed, so policy criteria can
quote it and retrieval can index it. Run once per LCD version:

    python -m priorauth.ingest path/to/LCD.pdf policies/sources/l38773_v32.txt
"""

import argparse
import re
import unicodedata
from pathlib import Path

SECTION_START = "Coverage Indications, Limitations, and/or Medical Necessity"
SECTION_END = "Summary of Evidence"
PRINT_DATE = re.compile(r"\d{1,2}/\d{1,2}/\d{2}, \d{1,2}:\d{2} [AP]M .*")  # browser header on every page
PRINT_URL = re.compile(r"(https://www\.cms\.gov/\S+) \d+/\d+")  # browser footer: source URL and page number
CURLY = str.maketrans({"\u2018": "'", "\u2019": "'", "\u201c": '"', "\u201d": '"'})


class IngestError(ValueError):
    """The PDF does not have the layout this extractor expects."""


def normalize(text: str) -> str:
    """Canonical form for checking that a quote appears in policy text.

    NFKC turns ligatures such as "ﬁ" into plain letters, curly quotes become straight ones, a word split
    by a hyphen at a line end is rejoined, and every run of whitespace becomes a single space.
    """
    text = unicodedata.normalize("NFKC", text).translate(CURLY)
    return " ".join(re.sub(r"-\n\s*", "-", text).split())


def page_body(page_text: str) -> tuple[list[str], str | None]:
    """Drop the browser's print header and footer; return the body lines and the source URL."""
    lines = page_text.splitlines()
    if lines and PRINT_DATE.fullmatch(lines[0]):
        lines = lines[1:]
    if lines and (footer := PRINT_URL.fullmatch(lines[-1])):
        return lines[:-1], footer.group(1)
    return lines, None


def coverage_section(lines: list[str]) -> list[str]:
    """Lines from the coverage heading up to, not including, the next section heading."""
    starts = [i for i, line in enumerate(lines) if line.startswith(SECTION_START)]
    if len(starts) != 1:
        raise IngestError(f"expected one {SECTION_START!r} heading, found {len(starts)}")
    ends = [i for i in range(starts[0] + 1, len(lines)) if lines[i].startswith(SECTION_END)]
    if not ends:
        raise IngestError(f"no {SECTION_END!r} heading after the coverage section")
    return lines[starts[0] : ends[0]]


def extract(page_texts: list[str]) -> str:
    """Build the committed text file from the text of each PDF page."""
    lines, urls = [], set()
    for page_text in page_texts:
        body, url = page_body(page_text)
        lines += body
        if url:
            urls.add(url)
    if len(urls) != 1:
        raise IngestError(f"expected one source URL in the page footers, found {len(urls)}")
    section = [unicodedata.normalize("NFKC", line).rstrip() for line in coverage_section(lines)]
    header = [f"# source: {urls.pop()}", f"# section: {SECTION_START}", ""]
    return "\n".join(header + section) + "\n"


def read_pdf(path: Path) -> list[str]:
    """Text of each page.

    y_tolerance=5 keeps emphasized words such as "ALL" in their sentence; with the default, pdfplumber
    moves them onto a line of their own.
    """
    import pdfplumber  # optional dependency: only needed when ingesting a new LCD version

    with pdfplumber.open(path) as pdf:
        return [page.extract_text(y_tolerance=5) or "" for page in pdf.pages]


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Extract the coverage section of an LCD PDF.")
    parser.add_argument("pdf", type=Path, help="LCD page printed to PDF")
    parser.add_argument("out", type=Path, help="text file to write, e.g. policies/sources/l38773_v32.txt")
    args = parser.parse_args(argv)
    text = extract(read_pdf(args.pdf))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(text)
    print(f"wrote {text.count(chr(10))} lines to {args.out}")


if __name__ == "__main__":
    main()