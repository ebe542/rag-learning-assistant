from uuid import UUID

import pytest

from rag_learning_assistant.chunking import Chunk
from rag_learning_assistant.retrieval import InMemoryVectorStore


def test_search_returns_most_similar_chunks_first() -> None:
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
    mixed_chunk = Chunk(
        text="Python database access",
        source="book.pdf",
        page_number=3,
        index=2,
    )

    store = InMemoryVectorStore()
    store.add(python_chunk, (1.0, 0.0))
    store.add(database_chunk, (0.0, 1.0))
    store.add(mixed_chunk, (0.8, 0.2))

    results = store.search((1.0, 0.0), limit=2)

    assert [result.chunk for result in results] == [
        python_chunk,
        mixed_chunk,
    ]
    assert results[0].score == 1.0
    assert results[0].score > results[1].score


@pytest.mark.parametrize("limit", [0, -1])
def test_search_limit_must_be_positive(limit: int) -> None:
    store = InMemoryVectorStore()

    with pytest.raises(ValueError, match="limit must be positive"):
        store.search((1.0, 0.0), limit=limit)


def make_chunk() -> Chunk:
    return Chunk(
        text="Content",
        source="book.pdf",
        page_number=1,
        index=0,
    )


@pytest.mark.parametrize("embedding", [(), (0.0, 0.0)])
def test_add_rejects_zero_vectors(embedding: tuple[float, ...]) -> None:
    store = InMemoryVectorStore()

    with pytest.raises(ValueError, match="Embedding must not be a zero vector"):
        store.add(make_chunk(), embedding)


@pytest.mark.parametrize("query", [(), (0.0, 0.0)])
def test_search_rejects_zero_query_vectors(query: tuple[float, ...]) -> None:
    store = InMemoryVectorStore()

    with pytest.raises(ValueError, match="Embedding must not be a zero vector"):
        store.search(query, limit=1)


def test_add_rejects_different_embedding_dimensions() -> None:
    store = InMemoryVectorStore()
    store.add(make_chunk(), (1.0, 0.0))

    with pytest.raises(ValueError, match="Embedding dimension must be 2"):
        store.add(make_chunk(), (1.0, 0.0, 0.0))


def test_search_rejects_different_query_dimension() -> None:
    store = InMemoryVectorStore()
    store.add(make_chunk(), (1.0, 0.0))

    with pytest.raises(ValueError, match="Embedding dimension must be 2"):
        store.search((1.0, 0.0, 0.0), limit=1)


def test_remove_document_keeps_entries_from_other_documents() -> None:
    removed_document_id = UUID("12345678-1234-5678-1234-567812345678")
    retained_document_id = UUID("87654321-4321-8765-4321-876543218765")
    removed_chunk = Chunk(
        text="Python functions",
        source="python.pdf",
        page_number=1,
        index=0,
        document_id=removed_document_id,
    )
    retained_chunk = Chunk(
        text="Relational databases",
        source="database.pdf",
        page_number=1,
        index=0,
        document_id=retained_document_id,
    )
    store = InMemoryVectorStore()
    store.add(removed_chunk, (1.0, 0.0))
    store.add(retained_chunk, (0.0, 1.0))

    removed_count = store.remove_document(removed_document_id)

    assert removed_count == 1
    assert [result.chunk for result in store.search((1.0, 0.0), limit=10)] == [retained_chunk]


def test_replace_document_keeps_other_entries() -> None:
    document_id = UUID("12345678-1234-5678-1234-567812345678")
    retained_document_id = UUID("87654321-4321-8765-4321-876543218765")
    old_chunk = Chunk(
        text="Old Python",
        source="old.pdf",
        page_number=1,
        index=0,
        document_id=document_id,
    )
    replacement_chunk = Chunk(
        text="Modern Python",
        source="new.pdf",
        page_number=1,
        index=0,
        document_id=document_id,
    )
    retained_chunk = Chunk(
        text="Databases",
        source="database.pdf",
        page_number=1,
        index=0,
        document_id=retained_document_id,
    )
    store = InMemoryVectorStore()
    store.add(old_chunk, (1.0, 0.0))
    store.add(retained_chunk, (0.0, 1.0))

    store.replace_document(
        document_id,
        [(replacement_chunk, (0.8, 0.2))],
    )

    assert [result.chunk for result in store.search((1.0, 0.0), limit=10)] == [
        replacement_chunk,
        retained_chunk,
    ]
