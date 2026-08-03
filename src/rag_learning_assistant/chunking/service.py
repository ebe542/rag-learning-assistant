"""Services for splitting document pages into searchable chunks."""

from collections.abc import Iterable

from rag_learning_assistant.chunking.models import Chunk
from rag_learning_assistant.ingestion import Page


class TextChunker:
    """Split document pages into searchable text chunks."""

    def __init__(self, max_chars: int, overlap_chars: int) -> None:
        if max_chars < 1:
            raise ValueError("max_chars must be positive")
        if overlap_chars < 0:
            raise ValueError("overlap_chars must not be negative")
        if overlap_chars >= max_chars:
            raise ValueError("overlap_chars must be smaller than max_chars")

        self.max_chars = max_chars
        self.overlap_chars = overlap_chars

    def chunk_pages(self, pages: Iterable[Page]) -> list[Chunk]:
        """Turn each non-empty page into one chunk."""

        chunks: list[Chunk] = []

        for page in pages:
            if not page.text.strip():
                continue

            for text in self._split_text(page.text):
                chunks.append(
                    Chunk(
                        text=text,
                        source=page.source,
                        page_number=page.number,
                        index=len(chunks),
                    )
                )

        return chunks

    def _split_text(self, text: str) -> list[str]:
        """Split text into overlapping chunks at word boundaries."""

        words = self._split_long_words(text)
        chunks: list[str] = []
        start = 0

        while start < len(words):
            end = start
            chunk_length = 0

            while end < len(words):
                separator_length = 1 if chunk_length else 0
                candidate_length = chunk_length + separator_length + len(words[end])

                if chunk_length and candidate_length > self.max_chars:
                    break

                chunk_length = candidate_length
                end += 1

            chunks.append(" ".join(words[start:end]))

            if end == len(words):
                break

            next_start = end
            overlap_length = 0

            while next_start > start:
                word = words[next_start - 1]
                separator_length = 1 if overlap_length else 0
                candidate_length = len(word) + separator_length + overlap_length

                if candidate_length > self.overlap_chars:
                    break

                overlap_length = candidate_length
                next_start -= 1

            # Ensure progress if the complete chunk fits into the overlap.
            start = end if next_start == start else next_start

        return chunks

    def _split_long_words(self, text: str) -> list[str]:
        """Split words that exceed the configured chunk size."""

        words: list[str] = []

        for word in text.split():
            parts = [
                word[start : start + self.max_chars]
                for start in range(0, len(word), self.max_chars)
            ]
            words.extend(parts)

        return words
