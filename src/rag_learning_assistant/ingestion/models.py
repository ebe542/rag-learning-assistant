"""Data models produced during document ingestion."""

from dataclasses import dataclass


def has_machine_readable_text(text: str) -> bool:
    """Return whether extracted text contains at least one Unicode letter."""

    return any(character.isalpha() for character in text)


@dataclass(frozen=True, slots=True)
class Page:
    """Text extracted from one document page."""

    number: int
    text: str
    source: str
    has_embedded_images: bool = False
    is_probable_full_page_scan: bool = False

    def __post_init__(self) -> None:
        if self.number < 1:
            raise ValueError("Page numbers start at 1")
        if not self.source:
            raise ValueError("A page needs a source")

    @property
    def has_machine_readable_text(self) -> bool:
        """Report whether this page can enter the text-processing pipeline."""

        return has_machine_readable_text(self.text)


@dataclass(frozen=True, slots=True)
class Document:
    """A document and its extracted pages."""

    source: str
    pages: tuple[Page, ...]

    @property
    def text(self) -> str:
        """Return non-empty pages separated by a blank line."""

        return "\n\n".join(page.text for page in self.pages if page.text)

    @property
    def pages_without_machine_readable_text(self) -> tuple[int, ...]:
        """Return stable OCR-candidate page numbers in document order."""

        return tuple(page.number for page in self.pages if not page.has_machine_readable_text)
