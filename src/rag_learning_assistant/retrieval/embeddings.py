"""Interfaces and types for text embeddings."""

from collections.abc import Sequence
from typing import Protocol, TypeAlias

Embedding: TypeAlias = tuple[float, ...]


class Embedder(Protocol):
    """Create embeddings for documents and search queries."""

    def embed_documents(
        self,
        texts: Sequence[str],
    ) -> list[Embedding]:
        """Return one passage embedding for every document text."""
        ...

    def embed_query(self, text: str) -> Embedding:
        """Return one query embedding for a search text."""
        ...
