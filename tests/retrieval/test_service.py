from collections.abc import Sequence

import pytest

from rag_learning_assistant.chunking import Chunk
from rag_learning_assistant.retrieval import (
    Embedding,
    InMemoryVectorStore,
    RetrievalService,
    SearchResult,
)


class EmptyEmbedder:
    """Return no document vectors regardless of the input."""

    def embed_documents(
        self,
        texts: Sequence[str],
    ) -> list[Embedding]:
        return []

    def embed_query(self, text: str) -> Embedding:
        return ()


class FakeEmbedder:
    """Return controlled vectors without loading a real ML model."""

    def __init__(self, embeddings: dict[str, Embedding]) -> None:
        self.embeddings = embeddings

    def embed_documents(
        self,
        texts: Sequence[str],
    ) -> list[Embedding]:
        return [self.embeddings[text] for text in texts]

    def embed_query(self, text: str) -> Embedding:
        return self.embeddings[text]


class RecordingVectorStore:
    """Record batch writes made by the retrieval service."""

    def __init__(self) -> None:
        self.batches: list[list[tuple[Chunk, Embedding]]] = []

    def add(self, chunk: Chunk, embedding: Embedding) -> None:
        raise AssertionError("RetrievalService must use a batch write")

    def add_many(
        self,
        entries: Sequence[tuple[Chunk, Embedding]],
    ) -> None:
        self.batches.append(list(entries))

    def search(
        self,
        query: Embedding,
        limit: int,
    ) -> list[SearchResult]:
        return []


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


def test_index_stores_all_embeddings_in_one_batch() -> None:
    first_chunk = Chunk(
        text="Python functions",
        source="book.pdf",
        page_number=1,
        index=0,
    )
    second_chunk = Chunk(
        text="Python classes",
        source="book.pdf",
        page_number=2,
        index=1,
    )
    embedder = FakeEmbedder(
        {
            "Python functions": (1.0, 0.0),
            "Python classes": (0.8, 0.2),
        }
    )
    store = RecordingVectorStore()
    service = RetrievalService(
        embedder=embedder,
        store=store,
    )

    service.index_chunks([first_chunk, second_chunk])

    assert store.batches == [
        [
            (first_chunk, (1.0, 0.0)),
            (second_chunk, (0.8, 0.2)),
        ]
    ]
