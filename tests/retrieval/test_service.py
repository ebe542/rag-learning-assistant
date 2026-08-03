from collections.abc import Sequence

import pytest

from rag_learning_assistant.chunking import Chunk
from rag_learning_assistant.retrieval import (
    Embedding,
    InMemoryVectorStore,
    RetrievalService,
)


class EmptyEmbedder:
    """Return no vectors regardless of the input."""

    def embed(self, texts: Sequence[str]) -> list[Embedding]:
        return []


class FakeEmbedder:
    """Return controlled vectors without loading a real ML model."""

    def __init__(self, embeddings: dict[str, Embedding]) -> None:
        self.embeddings = embeddings

    def embed(self, texts: Sequence[str]) -> list[Embedding]:
        return [self.embeddings[text] for text in texts]


class FixedEmbedder:
    """Return a fixed list of embeddings for every call."""

    def __init__(self, embeddings: list[Embedding]) -> None:
        self.embeddings = embeddings

    def embed(self, texts: Sequence[str]) -> list[Embedding]:
        return self.embeddings


def test_index_and_search_chunks() -> None:
    python_chunk = Chunk(
        text="Python functions",
        source="book.pdf",
        page_number=1,
        index=0,
    )
    database_chunk = Chunk(
        text="Relational databases",
        source="book.pdf",
        page_number=2,
        index=1,
    )
    embedder = FakeEmbedder(
        {
            "Python functions": (1.0, 0.0),
            "Relational databases": (0.0, 1.0),
            "How do functions work?": (1.0, 0.0),
        }
    )
    service = RetrievalService(
        embedder=embedder,
        store=InMemoryVectorStore(),
    )

    service.index_chunks([python_chunk, database_chunk])
    results = service.search("How do functions work?", limit=1)

    assert [result.chunk for result in results] == [python_chunk]
    assert results[0].score == 1.0


def test_index_rejects_wrong_number_of_embeddings() -> None:
    chunk = Chunk(
        text="Python functions",
        source="book.pdf",
        page_number=1,
        index=0,
    )
    service = RetrievalService(
        embedder=EmptyEmbedder(),
        store=InMemoryVectorStore(),
    )

    with pytest.raises(
        ValueError,
        match="Embedder must return one embedding per chunk",
    ):
        service.index_chunks([chunk])


@pytest.mark.parametrize(
    "embeddings",
    [
        [],
        [(1.0, 0.0), (0.0, 1.0)],
    ],
)
def test_search_requires_exactly_one_query_embedding(
    embeddings: list[Embedding],
) -> None:
    service = RetrievalService(
        embedder=FixedEmbedder(embeddings),
        store=InMemoryVectorStore(),
    )

    with pytest.raises(
        ValueError,
        match="Embedder must return exactly one query embedding",
    ):
        service.search("What is Python?", limit=1)
