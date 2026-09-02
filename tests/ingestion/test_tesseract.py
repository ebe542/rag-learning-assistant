from pathlib import Path

import pytest

from rag_learning_assistant.ingestion import TesseractPageOcr


class FakeOcrPage:
    def __init__(self, *, ocr_error: Exception | None = None) -> None:
        self.ocr_arguments: dict[str, object] | None = None
        self.text_arguments: tuple[str, object, bool] | None = None
        self.ocr_error = ocr_error

    def get_textpage_ocr(
        self,
        *,
        language: str,
        dpi: int,
        full: bool,
        tessdata: str | None,
    ) -> object:
        self.ocr_arguments = {
            "language": language,
            "dpi": dpi,
            "full": full,
            "tessdata": tessdata,
        }
        if self.ocr_error is not None:
            raise self.ocr_error
        return "ocr-text-page"

    def get_text(
        self,
        option: str = "text",
        *,
        textpage: object,
        sort: bool,
    ) -> str:
        self.text_arguments = (option, textpage, sort)
        return "Recognized text"


class FakeOcrDocument:
    def __init__(self, pages: list[FakeOcrPage]) -> None:
        self.pages = pages
        self.closed = False

    def __enter__(self) -> "FakeOcrDocument":
        return self

    def __exit__(self, *args: object) -> None:
        self.closed = True

    @property
    def page_count(self) -> int:
        return len(self.pages)

    def load_page(self, page_id: int) -> FakeOcrPage:
        return self.pages[page_id]


class StubTesseractPageOcr(TesseractPageOcr):
    def __init__(
        self,
        document: FakeOcrDocument,
        *,
        languages: str = "deu+eng",
        dpi: int = 300,
        tessdata: Path | None = None,
    ) -> None:
        super().__init__(languages=languages, dpi=dpi, tessdata=tessdata)
        self.document = document
        self.opened_path: Path | None = None

    def _open(self, path: Path) -> FakeOcrDocument:
        self.opened_path = path
        return self.document


def test_extract_text_uses_full_page_ocr_and_closes_document(tmp_path: Path) -> None:
    pdf = tmp_path / "scan.pdf"
    page = FakeOcrPage()
    document = FakeOcrDocument([page])
    ocr = StubTesseractPageOcr(
        document,
        languages="deu+eng",
        dpi=300,
        tessdata=tmp_path / "tessdata",
    )

    text = ocr.extract_text(pdf, 1)

    assert text == "Recognized text"
    assert ocr.opened_path == pdf
    assert page.ocr_arguments == {
        "language": "deu+eng",
        "dpi": 300,
        "full": True,
        "tessdata": str(tmp_path / "tessdata"),
    }
    assert page.text_arguments == ("text", "ocr-text-page", True)
    assert document.closed is True


def test_rejects_empty_languages() -> None:
    with pytest.raises(ValueError, match="languages must not be empty"):
        TesseractPageOcr(languages=" ")


def test_rejects_resolution_below_72_dpi() -> None:
    with pytest.raises(ValueError, match="resolution must be at least 72 DPI"):
        TesseractPageOcr(dpi=71)


def test_rejects_page_outside_document() -> None:
    ocr = StubTesseractPageOcr(FakeOcrDocument([FakeOcrPage()]))

    with pytest.raises(ValueError, match="OCR page 2 does not exist"):
        ocr.extract_text(Path("scan.pdf"), 2)


def test_reports_actionable_tesseract_configuration_error() -> None:
    page = FakeOcrPage(ocr_error=RuntimeError("source error"))
    ocr = StubTesseractPageOcr(FakeOcrDocument([page]))

    with pytest.raises(
        RuntimeError,
        match=r"verify TESSDATA_PREFIX.*deu\+eng",
    ):
        ocr.extract_text(Path("scan.pdf"), 1)
