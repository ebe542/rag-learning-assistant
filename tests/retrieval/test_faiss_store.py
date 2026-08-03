import sqlite3
from pathlib import Path

import pytest

from rag_learning_assistant.chunking import Chunk
from rag_learning_assistant.retrieval import FaissVectorStore


def test_entries_survive_reopening(tmp_path: Path) -> None:
    index_directory = tmp_path / "rag-index"
    chunk = Chunk(
        text="Python functions",
        source="book.pdf",
        page_number=3,
        index=7,
    )

    first_store = FaissVectorStore(
        index_directory,
        model_name="example/model",
        model_revision="revision-1",
    )
    first_store.add(chunk, (1.0, 0.0))

    reopened_store = FaissVectorStore(
        index_directory,
        model_name="example/model",
        model_revision="revision-1",
    )
    results = reopened_store.search((1.0, 0.0), limit=1)

    assert len(results) == 1
    assert results[0].chunk == chunk
    assert results[0].score == pytest.approx(1.0)


def test_search_preserves_faiss_ranking_after_reopening(tmp_path: Path) -> None:
    index_directory = tmp_path / "rag-index"
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

    store = FaissVectorStore(
        index_directory,
        model_name="example/model",
        model_revision="revision-1",
    )
    store.add(python_chunk, (1.0, 0.0))
    store.add(database_chunk, (0.0, 1.0))
    store.add(mixed_chunk, (0.8, 0.2))

    reopened_store = FaissVectorStore(
        index_directory,
        model_name="example/model",
        model_revision="revision-1",
    )
    results = reopened_store.search((1.0, 0.0), limit=2)

    assert [result.chunk for result in results] == [
        python_chunk,
        mixed_chunk,
    ]
    assert results[0].score == pytest.approx(1.0)
    assert results[0].score > results[1].score


def test_reopening_rejects_different_model_revision(tmp_path: Path) -> None:
    index_directory = tmp_path / "rag-index"
    chunk = Chunk(
        text="Python functions",
        source="book.pdf",
        page_number=1,
        index=0,
    )
    original_store = FaissVectorStore(
        index_directory,
        model_name="example/model",
        model_revision="revision-1",
    )
    original_store.add(chunk, (1.0, 0.0))

    with pytest.raises(
        ValueError,
        match="Index was created with a different embedding model",
    ):
        FaissVectorStore(
            index_directory,
            model_name="example/model",
            model_revision="revision-2",
        )


def test_first_embedding_persists_its_dimension(tmp_path: Path) -> None:
    index_directory = tmp_path / "rag-index"
    store = FaissVectorStore(
        index_directory,
        model_name="example/model",
        model_revision="revision-1",
    )
    chunk = Chunk(
        text="Python functions",
        source="book.pdf",
        page_number=1,
        index=0,
    )

    store.add(chunk, (1.0, 0.0, 0.0))

    with sqlite3.connect(index_directory / "metadata.sqlite3") as connection:
        stored_dimension = connection.execute(
            """
            SELECT embedding_dimension
            FROM index_metadata
            WHERE id = 1
            """
        ).fetchone()

    assert stored_dimension == (3,)


def test_add_many_persists_all_entries(tmp_path: Path) -> None:
    index_directory = tmp_path / "rag-index"
    first_chunk = Chunk(
        text="Python functions",
        source="book.pdf",
        page_number=1,
        index=0,
    )
    second_chunk = Chunk(
        text="Relational databases",
        source="book.pdf",
        page_number=2,
        index=1,
    )
    store = FaissVectorStore(
        index_directory,
        model_name="example/model",
        model_revision="revision-1",
    )

    store.add_many(
        [
            (first_chunk, (1.0, 0.0)),
            (second_chunk, (0.0, 1.0)),
        ]
    )

    reopened_store = FaissVectorStore(
        index_directory,
        model_name="example/model",
        model_revision="revision-1",
    )
    results = reopened_store.search((1.0, 0.0), limit=2)

    assert [result.chunk for result in results] == [
        first_chunk,
        second_chunk,
    ]
