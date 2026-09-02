"""Optional Tesseract OCR through PyMuPDF's native integration."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol, cast

DEFAULT_OCR_LANGUAGES = "deu+eng"


class OcrPdfPage(Protocol):
    """The PyMuPDF page operations required for OCR."""

    def get_textpage_ocr(
        self,
        *,
        language: str,
        dpi: int,
        full: bool,
        tessdata: str | None,
    ) -> object: ...

    def get_text(
        self,
        option: str = "text",
        *,
        textpage: object,
        sort: bool,
    ) -> str: ...


class OcrPdfHandle(Protocol):
    """The PyMuPDF document operations required for page OCR."""

    def __enter__(self) -> OcrPdfHandle: ...

    def __exit__(self, *args: object) -> None: ...

    @property
    def page_count(self) -> int: ...

    def load_page(self, page_id: int) -> OcrPdfPage: ...


class TesseractPageOcr:
    """Recognize complete scanned pages with Tesseract via PyMuPDF."""

    def __init__(
        self,
        *,
        languages: str = DEFAULT_OCR_LANGUAGES,
        dpi: int = 300,
        tessdata: Path | None = None,
    ) -> None:
        if not languages.strip():
            raise ValueError("OCR languages must not be empty")
        if dpi < 72:
            raise ValueError("OCR resolution must be at least 72 DPI")

        self.languages = languages
        self.dpi = dpi
        self.tessdata = tessdata

    def extract_text(self, path: Path, page_number: int) -> str:
        """OCR one complete page and return text in reading order."""

        if page_number < 1:
            raise ValueError("OCR page numbers start at 1")

        with self._open(path) as document:
            if page_number > document.page_count:
                raise ValueError(f"OCR page {page_number} does not exist")

            page = document.load_page(page_number - 1)
            try:
                text_page = page.get_textpage_ocr(
                    language=self.languages,
                    dpi=self.dpi,
                    full=True,
                    tessdata=str(self.tessdata) if self.tessdata is not None else None,
                )
            except Exception as error:
                raise RuntimeError(
                    "Tesseract OCR failed; verify TESSDATA_PREFIX and the installed "
                    f"language data for {self.languages}"
                ) from error
            return page.get_text("text", textpage=text_page, sort=True)

    def _open(self, path: Path) -> OcrPdfHandle:
        try:
            import pymupdf
        except ImportError as error:  # pragma: no cover - required base dependency
            raise RuntimeError("PyMuPDF is required for OCR") from error

        return cast(OcrPdfHandle, pymupdf.open(path))
