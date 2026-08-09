import hashlib
from pathlib import Path
from uuid import UUID

import pytest

from rag_learning_assistant.application import (
    DocumentNotFoundError,
    DuplicateDocumentError,
    LibraryService,
)
from rag_learning_assistant.chunking import Chunk
from rag_learning_assistant.ingestion import Document, Page
from rag_learning_assistant.library import IndexedDocument


class RecordingRepository:
    def __init__(
        self,
        existing_document: IndexedDocument,
        events: list[str],
        duplicate_document: IndexedDocument | None = None,
    ) -> None:
        self.existing_document = existing_document
        self.events = events
        self.updated_documents: list[IndexedDocument] = []
        self.duplicate_document = duplicate_document

    def add(self, document: IndexedDocument) -> None:
        raise AssertionError("add must not be called during document replacement")

    def list_all(self) -> list[IndexedDocument]:
        raise AssertionError("list_all must not be called during document replacement")

    def remove(self, document_id: UUID) -> None:
        raise AssertionError("remove must not be called during document replacement")

    def find_by_id(
        self,
        document_id: UUID,
    ) -> IndexedDocument | None:
        if document_id == self.existing_document.id:
            return self.existing_document

        return None

    def find_by_content_hash(
        self,
        content_sha256: str,
    ) -> IndexedDocument | None:
        if (
            self.duplicate_document is not None
            and self.duplicate_document.content_sha256 == content_sha256
        ):
            return self.duplicate_document

        return None

    def update(self, document: IndexedDocument) -> None:
        self.events.append("update metadata")
        self.updated_documents.append(document)


class FakeExtractor:
    def __init__(self, document: Document) -> None:
        self.document = document
        self.paths: list[Path] = []

    def extract(self, path: Path) -> Document:
        self.paths.append(path)
        return self.document


class RecordingIndexer:
    def __init__(
        self,
        chunks: list[Chunk],
        events: list[str],
    ) -> None:
        self.chunks = chunks
        self.events = events
        self.calls: list[tuple[Document, UUID]] = []

    def replace_document(
        self,
        document: Document,
        document_id: UUID,
    ) -> list[Chunk]:
        self.events.append("replace chunks")
        self.calls.append((document, document_id))
        return list(self.chunks)

    def index_document(
        self,
        document: Document,
        *,
        document_id: UUID | None = None,
    ) -> list[Chunk]:
        raise AssertionError("index_document must not be called during document replacement")

    def remove_document(self, document_id: UUID) -> int:
        raise AssertionError("remove_document must not be called during document replacement")


def test_replace_document_preserves_id_and_updates_metadata(
    tmp_path: Path,
) -> None:
    document_id = UUID("12345678-1234-5678-1234-567812345678")
    existing_document = IndexedDocument(
        id=document_id,
        source="old-book.pdf",
        content_sha256="a" * 64,
        page_count=5,
        chunk_count=12,
    )
    replacement_path = tmp_path / "new-book.pdf"
    replacement_path.write_bytes(b"new PDF contents")
    extracted_document = Document(
        source="new-book.pdf",
        pages=(
            Page(1, "New Python lesson", "new-book.pdf"),
            Page(2, "New database lesson", "new-book.pdf"),
        ),
    )
    replacement_chunks = [
        Chunk(
            text="New Python lesson",
            source="new-book.pdf",
            page_number=1,
            index=0,
            document_id=document_id,
        ),
        Chunk(
            text="New database lesson",
            source="new-book.pdf",
            page_number=2,
            index=1,
            document_id=document_id,
        ),
    ]
    events: list[str] = []
    repository = RecordingRepository(existing_document, events)
    extractor = FakeExtractor(extracted_document)
    indexer = RecordingIndexer(replacement_chunks, events)
    service = LibraryService(
        repository=repository,
        extractor=extractor,
        indexer=indexer,
    )

    replaced_document = service.replace_document(
        document_id,
        replacement_path,
    )

    expected_document = IndexedDocument(
        id=document_id,
        source="new-book.pdf",
        content_sha256=hashlib.sha256(b"new PDF contents").hexdigest(),
        page_count=2,
        chunk_count=2,
    )
    assert replaced_document == expected_document
    assert repository.updated_documents == [expected_document]
    assert extractor.paths == [replacement_path]
    assert indexer.calls == [(extracted_document, document_id)]
    assert events == [
        "replace chunks",
        "update metadata",
    ]


def test_replace_document_rejects_unknown_id(
    tmp_path: Path,
) -> None:
    existing_document = IndexedDocument(
        id=UUID("12345678-1234-5678-1234-567812345678"),
        source="old-book.pdf",
        content_sha256="a" * 64,
        page_count=5,
        chunk_count=12,
    )
    unknown_id = UUID("aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa")
    replacement_path = tmp_path / "new-book.pdf"
    replacement_path.write_bytes(b"new PDF contents")
    events: list[str] = []
    extractor = FakeExtractor(
        Document(
            source="new-book.pdf",
            pages=(Page(1, "New lesson", "new-book.pdf"),),
        )
    )
    indexer = RecordingIndexer([], events)
    service = LibraryService(
        repository=RecordingRepository(existing_document, events),
        extractor=extractor,
        indexer=indexer,
    )

    with pytest.raises(
        DocumentNotFoundError,
        match=str(unknown_id),
    ):
        service.replace_document(unknown_id, replacement_path)

    assert extractor.paths == []
    assert indexer.calls == []
    assert events == []


def test_replace_document_rejects_content_of_another_document(
    tmp_path: Path,
) -> None:
    document_id = UUID("12345678-1234-5678-1234-567812345678")
    existing_document = IndexedDocument(
        id=document_id,
        source="old-book.pdf",
        content_sha256="a" * 64,
        page_count=5,
        chunk_count=12,
    )
    replacement_path = tmp_path / "duplicate.pdf"
    replacement_path.write_bytes(b"duplicate contents")
    duplicate_document = IndexedDocument(
        id=UUID("87654321-4321-8765-4321-876543218765"),
        source="other-book.pdf",
        content_sha256=hashlib.sha256(b"duplicate contents").hexdigest(),
        page_count=3,
        chunk_count=6,
    )
    events: list[str] = []
    extractor = FakeExtractor(
        Document(
            source="duplicate.pdf",
            pages=(Page(1, "Duplicate", "duplicate.pdf"),),
        )
    )
    indexer = RecordingIndexer([], events)
    service = LibraryService(
        repository=RecordingRepository(
            existing_document,
            events,
            duplicate_document=duplicate_document,
        ),
        extractor=extractor,
        indexer=indexer,
    )

    with pytest.raises(
        DuplicateDocumentError,
        match="other-book.pdf",
    ):
        service.replace_document(document_id, replacement_path)

    assert extractor.paths == []
    assert indexer.calls == []
    assert events == []
