from collections.abc import Callable
from pathlib import Path

import pytest

from rag_learning_assistant.ingestion import Page, PdfExtractor


class FakePage:
    def __init__(self, text: str, *, error: Exception | None = None) -> None:
        self.text = text
        self.error = error

    def get_text(self, option: str = "text") -> str:
        assert option == "text"
        if self.error is not None:
            raise self.error
        return self.text


class FakePdf:
    def __init__(self, pages: list[FakePage], *, needs_pass: bool = False) -> None:
        self.pages = pages
        self.needs_pass = needs_pass

    def __enter__(self):
        return self

    def __exit__(self, *args: object) -> None:
        return None

    @property
    def page_count(self) -> int:
        return len(self.pages)

    def load_page(self, page_id: int) -> FakePage:
        return self.pages[page_id]


class FakeDiagnosticTools:
    def __init__(self, message: str) -> None:
        self.message = message
        self.errors_visible = True
        self.warnings_visible = True

    def mupdf_display_errors(
        self,
        value: bool | None = None,
    ) -> bool:
        previous = self.errors_visible
        if value is not None:
            self.errors_visible = value
        return previous

    def mupdf_display_warnings(
        self,
        value: bool | None = None,
    ) -> bool:
        previous = self.warnings_visible
        if value is not None:
            self.warnings_visible = value
        return previous

    def reset_mupdf_warnings(self) -> None:
        return None

    def mupdf_warnings(
        self,
        reset: bool = True,
    ) -> str:
        return self.message


class StubExtractor(PdfExtractor):
    def __init__(
        self,
        pages: list[FakePage],
        *,
        diagnostic_handler: Callable[[Path, str], None] | None = None,
        diagnostic_tools: FakeDiagnosticTools | None = None,
        needs_pass: bool = False,
    ) -> None:
        super().__init__(diagnostic_handler=diagnostic_handler)
        self.pages = pages
        self.diagnostic_tools = diagnostic_tools
        self.needs_pass = needs_pass

    def _open(self, path: Path) -> FakePdf:
        return FakePdf(self.pages, needs_pass=self.needs_pass)

    def _get_diagnostic_tools(
        self,
    ) -> FakeDiagnosticTools:
        if self.diagnostic_tools is None:
            raise AssertionError("No diagnostic tools configured")

        return self.diagnostic_tools


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


@pytest.mark.parametrize(
    "page_texts",
    [
        ["", "  \r\n"],
        ["\x00\x01", "1234 -- ???"],
    ],
)
def test_rejects_pdf_without_machine_readable_text(
    tmp_path: Path,
    page_texts: list[str],
) -> None:
    pdf = tmp_path / "scanned-or-empty.pdf"
    pdf.touch()
    extractor = StubExtractor([FakePage(text) for text in page_texts])

    with pytest.raises(ValueError, match="does not contain machine-readable text"):
        extractor.extract(pdf)


def test_accepts_pdf_with_a_single_extracted_letter(tmp_path: Path) -> None:
    pdf = tmp_path / "single-letter.pdf"
    pdf.touch()

    document = StubExtractor([FakePage("A")]).extract(pdf)

    assert document.text == "A"


def test_extract_removes_control_characters_from_readable_text(tmp_path: Path) -> None:
    pdf = tmp_path / "readable.pdf"
    pdf.touch()
    extractor = StubExtractor([FakePage("\x00Readable\x01\ttext\rnext line")])

    document = extractor.extract(pdf)

    assert document.text == "Readable\ttext\nnext line"


def test_extract_reports_unreadable_pages_in_readable_document(tmp_path: Path) -> None:
    pdf = tmp_path / "partly-readable.pdf"
    pdf.touch()
    extractor = StubExtractor(
        [FakePage("Readable first page"), FakePage("123 --"), FakePage("Final page")]
    )

    document = extractor.extract(pdf)

    assert document.pages_without_machine_readable_text == (2,)


def test_rejects_password_protected_pdf_before_reading_pages(tmp_path: Path) -> None:
    pdf = tmp_path / "protected.pdf"
    pdf.touch()
    extractor = StubExtractor(
        [FakePage("must not be read", error=AssertionError("page was read"))],
        needs_pass=True,
    )

    with pytest.raises(ValueError, match="PDF is password protected"):
        extractor.extract(pdf)


def test_rejects_pdf_without_pages(tmp_path: Path) -> None:
    pdf = tmp_path / "empty.pdf"
    pdf.touch()

    with pytest.raises(ValueError, match="PDF does not contain any pages"):
        StubExtractor([]).extract(pdf)


def test_reports_page_number_when_text_extraction_fails(tmp_path: Path) -> None:
    pdf = tmp_path / "broken-page.pdf"
    pdf.touch()
    extractor = StubExtractor(
        [FakePage("Readable first page"), FakePage("", error=RuntimeError("broken stream"))]
    )

    with pytest.raises(ValueError, match="Could not extract text from PDF page 2"):
        extractor.extract(pdf)


@pytest.mark.parametrize("filename", ["notes.txt", "book.epub", "pdf"])
def test_rejects_non_pdf_files(tmp_path: Path, filename: str) -> None:
    path = tmp_path / filename
    path.touch()

    with pytest.raises(ValueError, match="Expected a PDF"):
        PdfExtractor().extract(path)


def test_reports_missing_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        PdfExtractor().extract(tmp_path / "missing.pdf")


def test_extract_forwards_hidden_mupdf_diagnostics(
    tmp_path: Path,
) -> None:
    pdf = tmp_path / "document.pdf"
    pdf.touch()
    diagnostics: list[tuple[Path, str]] = []
    tools = FakeDiagnosticTools("cmsOpenProfileFromMem failed")
    extractor = StubExtractor(
        [FakePage("Document text")],
        diagnostic_handler=lambda path, message: diagnostics.append((path, message)),
        diagnostic_tools=tools,
    )

    document = extractor.extract(pdf)

    assert document.text == "Document text"
    assert diagnostics == [
        (
            pdf,
            "cmsOpenProfileFromMem failed",
        )
    ]
    assert tools.errors_visible is True
    assert tools.warnings_visible is True
