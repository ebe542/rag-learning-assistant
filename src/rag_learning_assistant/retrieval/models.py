"""Data models for retrieving relevant chunks."""

from dataclasses import dataclass

from rag_learning_assistant.chunking import Chunk


@dataclass(frozen=True, slots=True)
class SearchResult:
    """A retrieved chunk and its similarity score."""

    chunk: Chunk
    score: float
