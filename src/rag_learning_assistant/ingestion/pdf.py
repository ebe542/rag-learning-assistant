"""PDF ingestion with stable source metadata."""

from __future__ import annotations

import unicodedata
from collections.abc import Callable, Generator
from contextlib import contextmanager, suppress
from pathlib import Path
from typing import Protocol, cast

from rag_learning_assistant.ingestion.models import (
    Document,
    Page,
    has_machine_readable_text,
    has_usable_ocr_text,
)
from rag_learning_assistant.ingestion.ocr import PageOcr


class PdfPage(Protocol):
    """The subset of a PyMuPDF page used by the extractor."""

    def get_text(self, option: str = "text") -> str: ...

    def get_image_info(self) -> list[dict[str, object]]: ...

    @property
    def rect(self) -> PdfRect: ...


class PdfRect(Protocol):
    """The page rectangle coordinates needed for image coverage checks."""

    x0: float
    y0: float
    x1: float
    y1: float


class PdfHandle(Protocol):
    """A readable PDF handle, kept abstract for lightweight tests."""

    def __enter__(self) -> PdfHandle: ...

    def __exit__(self, *args: object) -> None: ...

    @property
    def page_count(self) -> int: ...

    @property
    def needs_pass(self) -> bool: ...

    def load_page(self, page_id: int) -> PdfPage: ...


class PdfDiagnosticTools(Protocol):
    """The subset of PyMuPDF diagnostics used by the extractor."""

    def mupdf_display_errors(
        self,
        value: bool | None = None,
    ) -> bool: ...

    def mupdf_display_warnings(
        self,
        value: bool | None = None,
    ) -> bool: ...

    def reset_mupdf_warnings(self) -> None: ...

    def mupdf_warnings(
        self,
        reset: bool = True,
    ) -> str: ...


class PdfExtractor:
    """Extract text page by page while preserving citation metadata."""

    def __init__(
        self,
        diagnostic_handler: Callable[[Path, str], None] | None = None,
        ocr: PageOcr | None = None,
    ) -> None:
        self.diagnostic_handler = diagnostic_handler
        self.ocr = ocr

    def extract(self, path: str | Path) -> Document:
        pdf_path = Path(path)
        self._validate(pdf_path)

        with (
            self._capture_diagnostics(pdf_path),
            self._open(pdf_path) as pdf,
        ):
            if pdf.needs_pass:
                raise ValueError("PDF is password protected")
            if pdf.page_count < 1:
                raise ValueError("PDF does not contain any pages")

            pages = tuple(
                self._extract_page(
                    pdf,
                    page_index,
                    path=pdf_path,
                    source=pdf_path.name,
                )
                for page_index in range(pdf.page_count)
            )

        document = Document(source=pdf_path.name, pages=pages)
        if not has_machine_readable_text(document.text) and not any(
            page.ocr_text_usable for page in document.pages
        ):
            if any(page.ocr_applied for page in document.pages):
                raise ValueError("OCR did not produce usable machine-readable text")
            raise ValueError("PDF does not contain machine-readable text")
        return document

    def _extract_page(
        self,
        pdf: PdfHandle,
        page_index: int,
        *,
        path: Path,
        source: str,
    ) -> Page:
        """Extract one page and identify its one-based location on failure."""

        page_number = page_index + 1
        try:
            pdf_page = pdf.load_page(page_index)
            raw_text = pdf_page.get_text("text")
            image_info = pdf_page.get_image_info()
            has_embedded_images = bool(image_info)
            is_probable_full_page_scan = self._is_probable_full_page_scan(
                pdf_page.rect,
                image_info,
            )
        except Exception as error:
            raise ValueError(f"Could not extract text from PDF page {page_number}") from error
        has_corrupt_text_mapping = self._has_corrupt_text_mapping(raw_text)
        text = self._normalise(raw_text)
        ocr_applied = False
        ocr_text_usable: bool | None = None
        if (
            not has_machine_readable_text(text)
            and (is_probable_full_page_scan or has_corrupt_text_mapping)
            and self.ocr is not None
        ):
            try:
                recognized_text = self._normalise(self.ocr.extract_text(path, page_number))
                ocr_applied = True
                ocr_text_usable = has_usable_ocr_text(recognized_text)
                text = recognized_text if ocr_text_usable else ""
            except Exception as error:
                raise ValueError(f"Could not OCR PDF page {page_number}: {error}") from error
        return Page(
            number=page_number,
            text=text,
            source=source,
            has_embedded_images=has_embedded_images,
            is_probable_full_page_scan=is_probable_full_page_scan,
            has_corrupt_text_mapping=has_corrupt_text_mapping,
            ocr_applied=ocr_applied,
            ocr_text_usable=ocr_text_usable,
        )

    @staticmethod
    def _is_probable_full_page_scan(
        page_rect: PdfRect,
        image_info: list[dict[str, object]],
    ) -> bool:
        """Return whether one image conservatively covers almost the entire page."""

        if len(image_info) != 1:
            return False

        bbox = image_info[0].get("bbox")
        if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
            return False

        image_x0, image_y0, image_x1, image_y1 = (float(value) for value in bbox)
        page_width = max(0.0, page_rect.x1 - page_rect.x0)
        page_height = max(0.0, page_rect.y1 - page_rect.y0)
        page_area = page_width * page_height
        if page_area == 0:
            return False

        covered_width = max(
            0.0,
            min(page_rect.x1, image_x1) - max(page_rect.x0, image_x0),
        )
        covered_height = max(
            0.0,
            min(page_rect.y1, image_y1) - max(page_rect.y0, image_y0),
        )
        return covered_width * covered_height / page_area >= 0.9

    @staticmethod
    def _has_corrupt_text_mapping(text: str) -> bool:
        """Detect substantial invalid control output from a broken PDF font map."""

        content = tuple(character for character in text if not character.isspace())
        if not content:
            return False

        invalid_controls = sum(
            character not in "\n\r\t" and unicodedata.category(character).startswith("C")
            for character in content
        )
        return invalid_controls >= 10 and invalid_controls / len(content) >= 0.5

    def _open(self, path: Path) -> PdfHandle:
        try:
            import pymupdf
        except ImportError as exc:  # pragma: no cover - depends on installation
            raise RuntimeError(
                "PyMuPDF is required for PDF extraction. Install the project first."
            ) from exc

        try:
            return cast(PdfHandle, pymupdf.open(path))
        except Exception as exc:
            raise ValueError(f"Could not open PDF: {path}") from exc

    @contextmanager
    def _capture_diagnostics(
        self,
        path: Path,
    ) -> Generator[None, None, None]:
        """Hide captured MuPDF messages and restore global settings."""

        if self.diagnostic_handler is None:
            yield
            return

        tools = self._get_diagnostic_tools()
        errors_visible = tools.mupdf_display_errors()
        warnings_visible = tools.mupdf_display_warnings()
        tools.reset_mupdf_warnings()
        tools.mupdf_display_errors(False)
        tools.mupdf_display_warnings(False)

        try:
            yield
        finally:
            message = tools.mupdf_warnings()
            tools.mupdf_display_errors(errors_visible)
            tools.mupdf_display_warnings(warnings_visible)

            if message.strip():
                # Diagnostic persistence must never mask extraction success or
                # replace the original PDF exception with a logging failure.
                with suppress(Exception):
                    self.diagnostic_handler(path, message)

    @staticmethod
    def _get_diagnostic_tools() -> PdfDiagnosticTools:
        """Load PyMuPDF diagnostics lazily with the PDF dependency."""

        try:
            import pymupdf
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError(
                "PyMuPDF is required for PDF extraction. Install the project first."
            ) from exc

        return cast(PdfDiagnosticTools, pymupdf.TOOLS)

    @staticmethod
    def _validate(path: Path) -> None:
        if not path.exists():
            raise FileNotFoundError(path)
        if not path.is_file():
            raise ValueError(f"Not a file: {path}")
        if path.suffix.lower() != ".pdf":
            raise ValueError(f"Expected a PDF file: {path}")

    @staticmethod
    def _normalise(text: str) -> str:
        normalized_newlines = text.replace("\r\n", "\n").replace("\r", "\n")
        cleaned = "".join(
            character
            for character in normalized_newlines
            if character in "\n\t" or not unicodedata.category(character).startswith("C")
        )
        lines = (line.rstrip() for line in cleaned.split("\n"))
        return "\n".join(lines).strip()
