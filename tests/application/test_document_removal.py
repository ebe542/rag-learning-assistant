from uuid import UUID

import pytest

from rag_learning_assistant.application import (
    DocumentNotFoundError,
    LibraryService,
)
from rag_learning_assistant.library import IndexedDocument


class RecordingRepository:
    def __init__(
        self,
        document: IndexedDocument,
        events: list[str],
    ) -> None:
        self.document = document
        self.events = events

    def find_by_id(self, document_id: UUID) -> IndexedDocument | None:
        if document_id == self.document.id:
            return self.document

        return None

    def remove(self, document_id: UUID) -> None:
        self.events.append("remove metadata")


class RecordingIndexer:
    def __init__(self, events: list[str]) -> None:
        self.events = events

    def remove_document(self, document_id: UUID) -> int:
        self.events.append("remove chunks")
        return 2


def test_remove_document_removes_chunks_before_metadata() -> None:
    document = IndexedDocument(
        id=UUID("12345678-1234-5678-1234-567812345678"),
        source="python-book.pdf",
        content_sha256="a" * 64,
        page_count=10,
        chunk_count=2,
    )
    events: list[str] = []
    service = LibraryService(
        repository=RecordingRepository(document, events),
        extractor=object(),
        indexer=RecordingIndexer(events),
    )

    removed_document = service.remove_document(document.id)

    assert removed_document == document
    assert events == [
        "remove chunks",
        "remove metadata",
    ]


def test_remove_document_rejects_unknown_id() -> None:
    existing_document = IndexedDocument(
        id=UUID("12345678-1234-5678-1234-567812345678"),
        source="python-book.pdf",
        content_sha256="a" * 64,
        page_count=10,
        chunk_count=2,
    )
    unknown_id = UUID("aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa")
    events: list[str] = []
    service = LibraryService(
        repository=RecordingRepository(existing_document, events),
        extractor=object(),
        indexer=RecordingIndexer(events),
    )

    with pytest.raises(
        DocumentNotFoundError,
        match=str(unknown_id),
    ):
        service.remove_document(unknown_id)

    assert events == []
