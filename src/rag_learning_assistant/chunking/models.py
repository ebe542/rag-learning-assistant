"""Data models for document chunks."""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Chunk:
    """A searchable text segment with source metadata."""

    text: str
    source: str
    page_number: int
    index: int

    def __post_init__(self) -> None:
        if not self.text.strip():
            raise ValueError("Chunk text must not be blank")

        if not self.source.strip():
            raise ValueError("Chunk source must not be blank")

        if self.page_number < 1:
            raise ValueError("Page numbers start at 1")

        if self.index < 0:
            raise ValueError("Chunk index must not be negative")
