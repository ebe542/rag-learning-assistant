import sqlite3
from pathlib import Path
from uuid import UUID

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


def test_document_id_survives_reopening(tmp_path: Path) -> None:
    index_directory = tmp_path / "rag-index"
    document_id = UUID("12345678-1234-5678-1234-567812345678")
    chunk = Chunk(
        text="Python functions",
        source="book.pdf",
        page_number=1,
        index=0,
        document_id=document_id,
    )
    store = FaissVectorStore(
        index_directory,
        model_name="example/model",
        model_revision="revision-1",
    )
    store.add(chunk, (1.0, 0.0))

    reopened_store = FaissVectorStore(
        index_directory,
        model_name="example/model",
        model_revision="revision-1",
    )
    results = reopened_store.search((1.0, 0.0), limit=1)

    assert results[0].chunk.document_id == document_id


def test_opening_old_database_adds_document_id_column(
    tmp_path: Path,
) -> None:
    index_directory = tmp_path / "rag-index"
    index_directory.mkdir()
    database_path = index_directory / "metadata.sqlite3"

    with sqlite3.connect(database_path) as connection:
        connection.execute(
            """
            CREATE TABLE chunks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                text TEXT NOT NULL,
                source TEXT NOT NULL,
                page_number INTEGER NOT NULL,
                chunk_index INTEGER NOT NULL
            )
            """
        )
        connection.execute(
            """
            INSERT INTO chunks (
                text,
                source,
                page_number,
                chunk_index
            )
            VALUES (?, ?, ?, ?)
            """,
            (
                "Python functions",
                "book.pdf",
                1,
                0,
            ),
        )

    FaissVectorStore(
        index_directory,
        model_name="example/model",
        model_revision="revision-1",
    )

    with sqlite3.connect(database_path) as connection:
        column_names = {row[1] for row in connection.execute("PRAGMA table_info(chunks)")}
        stored_document_id = connection.execute(
            """
            SELECT document_id
            FROM chunks
            WHERE id = 1
            """
        ).fetchone()

    assert "document_id" in column_names
    assert stored_document_id == (None,)
