"""In-memory storage and similarity search for embedded chunks."""

from __future__ import annotations

import math
from typing import Protocol

from rag_learning_assistant.chunking import Chunk
from rag_learning_assistant.retrieval.embeddings import Embedding
from rag_learning_assistant.retrieval.models import SearchResult


class InMemoryVectorStore:
    """Store embedded chunks and search them using cosine similarity."""

    def __init__(self) -> None:
        self._entries: list[tuple[Chunk, Embedding]] = []
        self._dimension: int | None = None

    def add(self, chunk: Chunk, embedding: Embedding) -> None:
        """Add an embedded chunk to the store."""

        self._validate_non_zero(embedding)
        self._validate_dimension(embedding)

        if self._dimension is None:
            self._dimension = len(embedding)

        self._entries.append((chunk, embedding))

    def search(self, query: Embedding, limit: int) -> list[SearchResult]:
        """Return the most similar chunks in descending score order."""

        if limit < 1:
            raise ValueError("limit must be positive")

        self._validate_non_zero(query)
        self._validate_dimension(query)

        results = [
            SearchResult(
                chunk=chunk,
                score=self._cosine_similarity(query, embedding),
            )
            for chunk, embedding in self._entries
        ]

        return sorted(
            results,
            key=lambda result: result.score,
            reverse=True,
        )[:limit]

    def _validate_dimension(self, embedding: Embedding) -> None:
        """Require the dimension established by the first stored vector."""

        if self._dimension is not None and len(embedding) != self._dimension:
            raise ValueError(f"Embedding dimension must be {self._dimension}")

    @staticmethod
    def _cosine_similarity(left: Embedding, right: Embedding) -> float:
        """Calculate the cosine similarity of two vectors."""

        dot_product = sum(a * b for a, b in zip(left, right, strict=True))
        left_length = math.sqrt(sum(value * value for value in left))
        right_length = math.sqrt(sum(value * value for value in right))

        return dot_product / (left_length * right_length)

    @staticmethod
    def _validate_non_zero(embedding: Embedding) -> None:
        """Reject vectors for which cosine similarity is undefined."""

        if not embedding or not any(value != 0.0 for value in embedding):
            raise ValueError("Embedding must not be a zero vector")


class VectorStore(Protocol):
    """Store embedded chunks and retrieve similar results."""

    def add(self, chunk: Chunk, embedding: Embedding) -> None:
        """Add an embedded chunk to the store."""
        ...

    def search(self, query: Embedding, limit: int) -> list[SearchResult]:
        """Return the most similar stored chunks."""
        ...
