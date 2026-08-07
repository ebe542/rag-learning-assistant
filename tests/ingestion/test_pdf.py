from pathlib import Path

import pytest

from rag_learning_assistant.ingestion import Page, PdfExtractor


class FakePage:
    def __init__(self, text: str) -> None:
        self.text = text

    def get_text(self, option: str = "text") -> str:
        assert option == "text"
        return self.text


class FakePdf:
    def __init__(self, pages: list[FakePage]) -> None:
        self.pages = pages

    def __enter__(self):
        return self

    def __exit__(self, *args: object) -> None:
        return None

    @property
    def page_count(self) -> int:
        return len(self.pages)

    def load_page(self, page_id: int) -> FakePage:
        return self.pages[page_id]


class StubExtractor(PdfExtractor):
    def __init__(self, pages: list[FakePage]) -> None:
        self.pages = pages

    def _open(self, path: Path) -> FakePdf:
        return FakePdf(self.pages)


def test_extract_preserves_page_numbers_and_source(tmp_path: Path) -> None:
    pdf = tmp_path / "python-book.pdf"
    pdf.touch()
    extractor = StubExtractor([FakePage(" First page  \r\n"), FakePage("Second page")])

    document = extractor.extract(pdf)

    assert document.source == "python-book.pdf"
    assert document.pages == (
        Page(number=1, text="First page", source="python-book.pdf"),
        Page(number=2, text="Second page", source="python-book.pdf"),
    )
    assert document.text == "First page\n\nSecond page"


@pytest.mark.parametrize("filename", ["notes.txt", "book.epub", "pdf"])
def test_rejects_non_pdf_files(tmp_path: Path, filename: str) -> None:
    path = tmp_path / filename
    path.touch()

    with pytest.raises(ValueError, match="Expected a PDF"):
        PdfExtractor().extract(path)


def test_reports_missing_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        PdfExtractor().extract(tmp_path / "missing.pdf")
