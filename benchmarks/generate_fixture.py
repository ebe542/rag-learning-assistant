"""Generate the redistributable summarization benchmark PDF."""

from hashlib import sha256
from pathlib import Path

import fitz

FIXTURE_DIRECTORY = Path(__file__).resolve().parent / "fixtures"
SOURCE_PATH = FIXTURE_DIRECTORY / "summarization-source.md"
OUTPUT_PATH = FIXTURE_DIRECTORY / "summarization-document.pdf"
PAGE_SEPARATOR = "<!-- PAGE BREAK -->"


def main() -> None:
    """Render one source section onto each fixed-size PDF page."""

    source = SOURCE_PATH.read_text(encoding="utf-8")
    sections = [section.strip() for section in source.split(PAGE_SEPARATOR)]
    if len(sections) != 10 or any(not section for section in sections):
        raise ValueError("Benchmark source must contain exactly ten non-empty pages")

    document = fitz.open()
    try:
        document.set_metadata(
            {
                "title": "RAG Learning Assistant Summarization Benchmark",
                "author": "RAG Learning Assistant contributors",
                "subject": "Redistributable synthetic benchmark fixture",
                "keywords": "RAG, summarization, benchmark, synthetic",
                "creator": "benchmarks/generate_fixture.py",
                "producer": "PyMuPDF",
                "creationDate": "D:20260812000000+00'00'",
                "modDate": "D:20260812000000+00'00'",
            }
        )

        for section in sections:
            page = document.new_page(width=595, height=842)
            remaining_height = page.insert_textbox(
                fitz.Rect(54, 54, 541, 788),
                section,
                fontname="helv",
                fontsize=10,
                lineheight=1.25,
            )
            if remaining_height < 0:
                raise ValueError("Benchmark source does not fit on its assigned page")

        document.save(
            OUTPUT_PATH,
            garbage=4,
            deflate=True,
            no_new_id=True,
        )
    finally:
        document.close()

    digest = sha256(OUTPUT_PATH.read_bytes()).hexdigest()
    print(f"Created {OUTPUT_PATH}")
    print(f"SHA-256: {digest}")


if __name__ == "__main__":
    main()
