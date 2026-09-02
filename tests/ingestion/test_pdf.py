from collections.abc import Callable
from pathlib import Path

import pytest

from rag_learning_assistant.ingestion import Page, PdfExtractor


class FakeRect:
    x0 = 0.0
    y0 = 0.0
    x1 = 100.0
    y1 = 100.0


class FakePage:
    def __init__(
        self,
        text: str,
        *,
        error: Exception | None = None,
        has_embedded_images: bool = False,
        image_bbox: tuple[float, float, float, float] = (0.0, 0.0, 100.0, 100.0),
        image_count: int = 1,
    ) -> None:
        self.text = text
        self.error = error
        self.has_embedded_images = has_embedded_images
        self.image_bbox = image_bbox
        self.image_count = image_count
        self.rect = FakeRect()

    def get_text(self, option: str = "text") -> str:
        assert option == "text"
        if self.error is not None:
            raise self.error
        return self.text

    def get_image_info(self) -> list[dict[str, object]]:
        if not self.has_embedded_images:
            return []
        return [{"bbox": self.image_bbox} for _ in range(self.image_count)]


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


class FakeOcr:
    def __init__(self, results: dict[int, str | Exception]) -> None:
        self.results = results
        self.calls: list[tuple[Path, int]] = []

    def extract_text(self, path: Path, page_number: int) -> str:
        self.calls.append((path, page_number))
        result = self.results[page_number]
        if isinstance(result, Exception):
            raise result
        return result


class StubExtractor(PdfExtractor):
    def __init__(
        self,
        pages: list[FakePage],
        *,
        diagnostic_handler: Callable[[Path, str], None] | None = None,
        diagnostic_tools: FakeDiagnosticTools | None = None,
        needs_pass: bool = False,
        ocr: FakeOcr | None = None,
    ) -> None:
        super().__init__(diagnostic_handler=diagnostic_handler, ocr=ocr)
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


def test_extract_uses_ocr_only_for_pages_without_readable_text(tmp_path: Path) -> None:
    pdf = tmp_path / "partly-scanned.pdf"
    pdf.touch()
    ocr = FakeOcr({2: " OCR result\r\n"})
    extractor = StubExtractor(
        [FakePage("Native text"), FakePage("123 --", has_embedded_images=True)],
        ocr=ocr,
    )

    document = extractor.extract(pdf)

    assert document.pages == (
        Page(number=1, text="Native text", source=pdf.name),
        Page(
            number=2,
            text="OCR result",
            source=pdf.name,
            has_embedded_images=True,
            is_probable_full_page_scan=True,
        ),
    )
    assert document.pages_without_machine_readable_text == ()
    assert ocr.calls == [(pdf, 2)]


def test_extract_reports_page_number_when_ocr_fails(tmp_path: Path) -> None:
    pdf = tmp_path / "broken-scan.pdf"
    pdf.touch()
    extractor = StubExtractor(
        [FakePage("Native text"), FakePage("", has_embedded_images=True)],
        ocr=FakeOcr({2: RuntimeError("OCR backend failed")}),
    )

    with pytest.raises(ValueError, match="Could not OCR PDF page 2"):
        extractor.extract(pdf)


def test_extract_does_not_ocr_an_empty_page_without_images(tmp_path: Path) -> None:
    pdf = tmp_path / "document-with-empty-page.pdf"
    pdf.touch()
    ocr = FakeOcr({2: "must not be used"})
    extractor = StubExtractor([FakePage("Native text"), FakePage("")], ocr=ocr)

    document = extractor.extract(pdf)

    assert document.pages_without_machine_readable_text == (2,)
    assert document.pages[1].has_embedded_images is False
    assert ocr.calls == []


def test_extract_does_not_ocr_a_small_embedded_image(tmp_path: Path) -> None:
    pdf = tmp_path / "page-with-illustration.pdf"
    pdf.touch()
    ocr = FakeOcr({2: "must not be used"})
    extractor = StubExtractor(
        [
            FakePage("Native text"),
            FakePage(
                "",
                has_embedded_images=True,
                image_bbox=(20.0, 20.0, 80.0, 80.0),
            ),
        ],
        ocr=ocr,
    )

    document = extractor.extract(pdf)

    assert document.pages[1].has_embedded_images is True
    assert document.pages[1].is_probable_full_page_scan is False
    assert ocr.calls == []


def test_extract_does_not_ocr_multiple_full_page_images(tmp_path: Path) -> None:
    pdf = tmp_path / "page-with-multiple-images.pdf"
    pdf.touch()
    ocr = FakeOcr({2: "must not be used"})
    extractor = StubExtractor(
        [
            FakePage("Native text"),
            FakePage("", has_embedded_images=True, image_count=2),
        ],
        ocr=ocr,
    )

    document = extractor.extract(pdf)

    assert document.pages[1].is_probable_full_page_scan is False
    assert ocr.calls == []


def test_extract_uses_ocr_for_a_corrupt_font_mapping(tmp_path: Path) -> None:
    pdf = tmp_path / "broken-font-map.pdf"
    pdf.touch()
    ocr = FakeOcr({1: "Recovered text"})
    extractor = StubExtractor([FakePage("\x01\x02\x03\x04\x05" * 3)], ocr=ocr)

    document = extractor.extract(pdf)

    assert document.pages[0].text == "Recovered text"
    assert document.pages[0].has_corrupt_text_mapping is True
    assert ocr.calls == [(pdf, 1)]


def test_extract_does_not_ocr_a_symbol_only_page(tmp_path: Path) -> None:
    pdf = tmp_path / "symbols.pdf"
    pdf.touch()
    ocr = FakeOcr({1: "must not be used"})
    extractor = StubExtractor([FakePage("123 -- ???")], ocr=ocr)

    with pytest.raises(ValueError, match="does not contain machine-readable text"):
        extractor.extract(pdf)

    assert ocr.calls == []


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
