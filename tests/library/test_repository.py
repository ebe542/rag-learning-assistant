from pathlib import Path
from uuid import UUID

from rag_learning_assistant.library import (
    IndexedDocument,
    SqliteDocumentRepository,
)


def test_documents_survive_reopening(tmp_path: Path) -> None:
    database_path = tmp_path / "metadata.sqlite3"
    document = IndexedDocument(
        id=UUID("12345678-1234-5678-1234-567812345678"),
        source="python-book.pdf",
        content_sha256="a" * 64,
        page_count=120,
        chunk_count=758,
    )

    first_repository = SqliteDocumentRepository(database_path)
    first_repository.add(document)

    reopened_repository = SqliteDocumentRepository(database_path)

    assert reopened_repository.list_all() == [document]


def test_find_by_content_hash(tmp_path: Path) -> None:
    repository = SqliteDocumentRepository(tmp_path / "metadata.sqlite3")
    document = IndexedDocument(
        id=UUID("12345678-1234-5678-1234-567812345678"),
        source="python-book.pdf",
        content_sha256="a" * 64,
        page_count=120,
        chunk_count=758,
    )
    repository.add(document)

    assert repository.find_by_content_hash("a" * 64) == document
    assert repository.find_by_content_hash("b" * 64) is None


def test_remove_document_by_id_survives_reopening(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "metadata.sqlite3"
    removed_document = IndexedDocument(
        id=UUID("12345678-1234-5678-1234-567812345678"),
        source="python-book.pdf",
        content_sha256="a" * 64,
        page_count=10,
        chunk_count=20,
    )
    retained_document = IndexedDocument(
        id=UUID("87654321-4321-8765-4321-876543218765"),
        source="database-book.pdf",
        content_sha256="b" * 64,
        page_count=5,
        chunk_count=8,
    )
    repository = SqliteDocumentRepository(database_path)
    repository.add(removed_document)
    repository.add(retained_document)

    assert repository.find_by_id(removed_document.id) == removed_document

    repository.remove(removed_document.id)

    reopened_repository = SqliteDocumentRepository(database_path)

    assert reopened_repository.find_by_id(removed_document.id) is None
    assert reopened_repository.list_all() == [retained_document]
