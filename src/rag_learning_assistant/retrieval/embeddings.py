"""Interfaces and types for text embeddings."""

from collections.abc import Sequence
from typing import Protocol, TypeAlias

Embedding: TypeAlias = tuple[float, ...]


class Embedder(Protocol):
    """Convert batches of text into numeric vectors."""

    def embed(self, texts: Sequence[str]) -> list[Embedding]:
        """Return one embedding for every input text."""
        ...
