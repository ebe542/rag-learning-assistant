from pathlib import Path
from uuid import UUID

from rag_learning_assistant.chunking import Chunk
from rag_learning_assistant.retrieval import FaissVectorStore


def test_remove_document_keeps_chunks_from_other_documents(
    tmp_path: Path,
) -> None:
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
    store = FaissVectorStore(
        tmp_path / "rag-index",
        model_name="example/model",
        model_revision="revision-1",
    )
    store.add_many(
        [
            (removed_chunk, (1.0, 0.0)),
            (retained_chunk, (0.0, 1.0)),
        ]
    )

    removed_count = store.remove_document(removed_document_id)

    assert removed_count == 1
    assert [result.chunk for result in store.search((1.0, 0.0), limit=10)] == [retained_chunk]
