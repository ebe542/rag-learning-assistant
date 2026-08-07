"""PDF ingestion with stable source metadata."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol, cast

from rag_learning_assistant.ingestion.models import Document, Page


class PdfPage(Protocol):
    """The subset of a PyMuPDF page used by the extractor."""

    def get_text(self, option: str = "text") -> str: ...


class PdfHandle(Protocol):
    """A readable PDF handle, kept abstract for lightweight tests."""

    def __enter__(self) -> PdfHandle: ...

    def __exit__(self, *args: object) -> None: ...

    @property
    def page_count(self) -> int: ...

    def load_page(self, page_id: int) -> PdfPage: ...


class PdfExtractor:
    """Extract text page by page while preserving citation metadata."""

    def extract(self, path: str | Path) -> Document:
        pdf_path = Path(path)
        self._validate(pdf_path)

        with self._open(pdf_path) as pdf:
            pages = tuple(
                Page(
                    number=page_index + 1,
                    text=self._normalise(pdf.load_page(page_index).get_text("text")),
                    source=pdf_path.name,
                )
                for page_index in range(pdf.page_count)
            )

        return Document(source=pdf_path.name, pages=pages)

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
        lines = (line.rstrip() for line in text.replace("\r\n", "\n").split("\n"))
        return "\n".join(lines).strip()
