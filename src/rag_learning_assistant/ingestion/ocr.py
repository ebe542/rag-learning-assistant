"""OCR boundary used by document ingestion."""

from pathlib import Path
from typing import Protocol


class PageOcr(Protocol):
    """Extract text from one PDF page using an optional OCR backend."""

    def extract_text(self, path: Path, page_number: int) -> str:
        """Return text for a one-based page number."""

        ...
