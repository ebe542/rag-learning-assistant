"""Data models for documents stored in a library."""

from dataclasses import dataclass
from uuid import UUID


@dataclass(frozen=True, slots=True)
class IndexedDocument:
    """Metadata describing one document in a persistent library."""

    id: UUID
    source: str
    content_sha256: str
    page_count: int
    chunk_count: int

    def __post_init__(self) -> None:
        if not self.source.strip():
            raise ValueError("Document source must not be blank")

        if len(self.content_sha256) != 64 or any(
            character not in "0123456789abcdef" for character in self.content_sha256
        ):
            raise ValueError("Content hash must be a SHA-256 hexadecimal value")

        if self.page_count < 1:
            raise ValueError("Page count must be positive")

        if self.chunk_count < 0:
            raise ValueError("Chunk count must not be negative")
