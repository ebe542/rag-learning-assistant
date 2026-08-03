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
