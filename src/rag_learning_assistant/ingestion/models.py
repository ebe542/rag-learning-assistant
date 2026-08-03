"""Data models produced during document ingestion."""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Page:
    """Text extracted from one document page."""

    number: int
    text: str
    source: str

    def __post_init__(self) -> None:
        if self.number < 1:
            raise ValueError("Page numbers start at 1")
        if not self.source:
            raise ValueError("A page needs a source")


@dataclass(frozen=True, slots=True)
class Document:
    """A document and its extracted pages."""

    source: str
    pages: tuple[Page, ...]

    @property
    def text(self) -> str:
        """Return non-empty pages separated by a blank line."""

        return "\n\n".join(page.text for page in self.pages if page.text)
