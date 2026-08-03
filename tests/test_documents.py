from pathlib import Path

import pytest

from rag_learning_assistant.documents import Document, Page, PdfExtractor


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

    def __iter__(self):
        return iter(self.pages)


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


def test_document_text_ignores_empty_pages() -> None:
    document = Document(
        source="book.pdf",
        pages=(Page(1, "", "book.pdf"), Page(2, "Content", "book.pdf")),
    )

    assert document.text == "Content"


@pytest.mark.parametrize("filename", ["notes.txt", "book.epub", "pdf"])
def test_rejects_non_pdf_files(tmp_path: Path, filename: str) -> None:
    path = tmp_path / filename
    path.touch()

    with pytest.raises(ValueError, match="Expected a PDF"):
        PdfExtractor().extract(path)


def test_reports_missing_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        PdfExtractor().extract(tmp_path / "missing.pdf")


def test_page_numbers_must_be_positive() -> None:
    with pytest.raises(ValueError, match="start at 1"):
        Page(number=0, text="text", source="book.pdf")
