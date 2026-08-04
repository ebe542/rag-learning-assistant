from pathlib import Path
from uuid import UUID

import pytest

from rag_learning_assistant.chunking import Chunk
from rag_learning_assistant.retrieval import FaissVectorStore


def test_replace_document_preserves_other_documents_and_survives_reopening(
    tmp_path: Path,
) -> None:
    index_directory = tmp_path / "rag-index"
    replaced_document_id = UUID("12345678-1234-5678-1234-567812345678")
    retained_document_id = UUID("87654321-4321-8765-4321-876543218765")
    old_chunks = [
        Chunk(
            text="Old Python functions",
            source="old-book.pdf",
            page_number=1,
            index=0,
            document_id=replaced_document_id,
        ),
        Chunk(
            text="Old Python classes",
            source="old-book.pdf",
            page_number=2,
            index=1,
            document_id=replaced_document_id,
        ),
    ]
    retained_chunk = Chunk(
        text="Relational databases",
        source="database-book.pdf",
        page_number=1,
        index=0,
        document_id=retained_document_id,
    )
    replacement_chunk = Chunk(
        text="Modern Python functions",
        source="new-book.pdf",
        page_number=1,
        index=0,
        document_id=replaced_document_id,
    )
    store = FaissVectorStore(
        index_directory,
        model_name="example/model",
        model_revision="revision-1",
    )
    store.add_many(
        [
            (old_chunks[0], (1.0, 0.0)),
            (old_chunks[1], (0.9, 0.1)),
            (retained_chunk, (0.0, 1.0)),
        ]
    )

    store.replace_document(
        replaced_document_id,
        [(replacement_chunk, (1.0, 0.0))],
    )

    reopened_store = FaissVectorStore(
        index_directory,
        model_name="example/model",
        model_revision="revision-1",
    )
    results = reopened_store.search((1.0, 0.0), limit=10)

    assert [result.chunk for result in results] == [
        replacement_chunk,
        retained_chunk,
    ]


def test_failed_replacement_preserves_existing_document(
    tmp_path: Path,
) -> None:
    index_directory = tmp_path / "rag-index"
    document_id = UUID("12345678-1234-5678-1234-567812345678")
    old_chunk = Chunk(
        text="Old Python functions",
        source="old-book.pdf",
        page_number=1,
        index=0,
        document_id=document_id,
    )
    invalid_replacement = Chunk(
        text="Invalid replacement",
        source="new-book.pdf",
        page_number=1,
        index=0,
        document_id=document_id,
    )
    store = FaissVectorStore(
        index_directory,
        model_name="example/model",
        model_revision="revision-1",
    )
    store.add(old_chunk, (1.0, 0.0))

    with pytest.raises(
        ValueError,
        match="Embedding must not be a zero vector",
    ):
        store.replace_document(
            document_id,
            [(invalid_replacement, (0.0, 0.0))],
        )

    reopened_store = FaissVectorStore(
        index_directory,
        model_name="example/model",
        model_revision="revision-1",
    )
    results = reopened_store.search((1.0, 0.0), limit=10)

    assert [result.chunk for result in results] == [old_chunk]
