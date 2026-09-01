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
)


class PdfPage(Protocol):
    """The subset of a PyMuPDF page used by the extractor."""

    def get_text(self, option: str = "text") -> str: ...


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
    ) -> None:
        self.diagnostic_handler = diagnostic_handler

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
                self._extract_page(pdf, page_index, source=pdf_path.name)
                for page_index in range(pdf.page_count)
            )

        document = Document(source=pdf_path.name, pages=pages)
        if not has_machine_readable_text(document.text):
            raise ValueError("PDF does not contain machine-readable text")
        return document

    def _extract_page(self, pdf: PdfHandle, page_index: int, *, source: str) -> Page:
        """Extract one page and identify its one-based location on failure."""

        page_number = page_index + 1
        try:
            text = pdf.load_page(page_index).get_text("text")
        except Exception as error:
            raise ValueError(f"Could not extract text from PDF page {page_number}") from error
        return Page(
            number=page_number,
            text=self._normalise(text),
            source=source,
        )

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
