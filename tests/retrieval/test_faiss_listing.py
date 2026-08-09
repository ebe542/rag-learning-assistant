from pathlib import Path
from uuid import UUID

from rag_learning_assistant.chunking import Chunk
from rag_learning_assistant.retrieval import FaissVectorStore


def test_list_document_chunks_returns_only_requested_document_in_index_order(
    tmp_path: Path,
) -> None:
    document_id = UUID("12345678-1234-5678-1234-567812345678")
    other_document_id = UUID("87654321-4321-8765-4321-876543218765")

    third_chunk = Chunk(
        text="Python classes",
        source="python-book.pdf",
        page_number=3,
        index=2,
        document_id=document_id,
    )
    first_chunk = Chunk(
        text="Python basics",
        source="python-book.pdf",
        page_number=1,
        index=0,
        document_id=document_id,
    )
    second_chunk = Chunk(
        text="Python functions",
        source="python-book.pdf",
        page_number=2,
        index=1,
        document_id=document_id,
    )
    unrelated_chunk = Chunk(
        text="Relational databases",
        source="database-book.pdf",
        page_number=1,
        index=0,
        document_id=other_document_id,
    )

    store = FaissVectorStore(
        tmp_path / "rag-index",
        model_name="example/model",
        model_revision="revision-1",
    )
    store.add_many(
        [
            (third_chunk, (1.0, 0.0)),
            (unrelated_chunk, (0.0, 1.0)),
            (first_chunk, (0.8, 0.2)),
            (second_chunk, (0.6, 0.4)),
        ]
    )

    chunks = store.list_document_chunks(document_id)

    assert chunks == [
        first_chunk,
        second_chunk,
        third_chunk,
    ]


def test_list_document_chunks_returns_empty_list_for_unknown_document(
    tmp_path: Path,
) -> None:
    store = FaissVectorStore(
        tmp_path,
        model_name="test-model",
        model_revision="test-revision",
    )

    chunks = store.list_document_chunks(UUID("aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa"))

    assert chunks == []
