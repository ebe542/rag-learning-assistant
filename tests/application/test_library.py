import hashlib
from collections.abc import Sequence
from pathlib import Path
from uuid import UUID

import pytest

from rag_learning_assistant.application import (
    DuplicateDocumentError,
    LibraryCatalog,
    LibraryService,
)
from rag_learning_assistant.chunking import Chunk
from rag_learning_assistant.ingestion import Document, Page
from rag_learning_assistant.library import IndexedDocument
from rag_learning_assistant.retrieval import SearchResult


class RecordingDocumentRepository:
    def __init__(self) -> None:
        self.documents: list[IndexedDocument] = []

    def add(self, document: IndexedDocument) -> None:
        self.documents.append(document)

    def list_all(self) -> list[IndexedDocument]:
        return list(self.documents)

    def find_by_content_hash(
        self,
        content_sha256: str,
    ) -> IndexedDocument | None:
        return next(
            (document for document in self.documents if document.content_sha256 == content_sha256),
            None,
        )

    def find_by_id(self, document_id: UUID) -> IndexedDocument | None:
        raise AssertionError("find_by_id must not be called while adding or listing documents")

    def remove(self, document_id: UUID) -> None:
        raise AssertionError("remove must not be called while adding or listing documents")

    def update(self, document: IndexedDocument) -> None:
        raise AssertionError("update must not be called while adding or listing documents")


class FakeExtractor:
    def __init__(self, document: Document) -> None:
        self.document = document
        self.paths: list[Path] = []

    def extract(self, path: Path) -> Document:
        self.paths.append(path)
        return self.document


class FakeDocumentIndexer:
    def __init__(self, chunks: Sequence[Chunk]) -> None:
        self.chunks = list(chunks)
        self.indexed_documents: list[tuple[Document, UUID | None]] = []

    def index_document(
        self,
        document: Document,
        *,
        document_id: UUID | None = None,
    ) -> list[Chunk]:
        self.indexed_documents.append((document, document_id))
        return list(self.chunks)

    def remove_document(self, document_id: UUID) -> int:
        raise AssertionError("remove_document must not be called while adding documents")

    def replace_document(
        self,
        document: Document,
        document_id: UUID,
    ) -> list[Chunk]:
        raise AssertionError("replace_document must not be called while adding documents")

    def search(self, query: str, limit: int) -> list[SearchResult]:
        return []


def test_add_document_indexes_and_registers_metadata(
    tmp_path: Path,
) -> None:
    pdf = tmp_path / "course.pdf"
    pdf.write_bytes(b"PDF contents")
    document = Document(
        source="course.pdf",
        pages=(
            Page(1, "Python functions", "course.pdf"),
            Page(2, "Python classes", "course.pdf"),
        ),
    )
    chunks = [
        Chunk(
            text="Python functions",
            source="course.pdf",
            page_number=1,
            index=0,
        ),
        Chunk(
            text="Python classes",
            source="course.pdf",
            page_number=2,
            index=1,
        ),
    ]
    repository = RecordingDocumentRepository()
    extractor = FakeExtractor(document)
    indexer = FakeDocumentIndexer(chunks)
    service = LibraryService(
        repository=repository,
        extractor=extractor,
        indexer=indexer,
        id_factory=lambda: UUID("12345678-1234-5678-1234-567812345678"),
    )

    indexed_document = service.add_document(pdf)

    expected = IndexedDocument(
        id=UUID("12345678-1234-5678-1234-567812345678"),
        source="course.pdf",
        content_sha256=hashlib.sha256(b"PDF contents").hexdigest(),
        page_count=2,
        chunk_count=2,
    )
    assert indexed_document == expected
    assert repository.documents == [expected]
    assert extractor.paths == [pdf]
    assert indexer.indexed_documents == [
        (
            document,
            UUID("12345678-1234-5678-1234-567812345678"),
        )
    ]


def test_add_document_rejects_duplicate_content_before_indexing(
    tmp_path: Path,
) -> None:
    pdf = tmp_path / "renamed-course.pdf"
    pdf.write_bytes(b"PDF contents")
    content_sha256 = hashlib.sha256(b"PDF contents").hexdigest()
    existing_document = IndexedDocument(
        id=UUID("12345678-1234-5678-1234-567812345678"),
        source="course.pdf",
        content_sha256=content_sha256,
        page_count=2,
        chunk_count=2,
    )
    extracted_document = Document(
        source="renamed-course.pdf",
        pages=(Page(1, "Python functions", "renamed-course.pdf"),),
    )
    repository = RecordingDocumentRepository()
    repository.documents.append(existing_document)
    extractor = FakeExtractor(extracted_document)
    indexer = FakeDocumentIndexer([])
    service = LibraryService(
        repository=repository,
        extractor=extractor,
        indexer=indexer,
    )

    with pytest.raises(
        DuplicateDocumentError,
        match="Document content is already indexed",
    ):
        service.add_document(pdf)

    assert repository.documents == [existing_document]
    assert extractor.paths == []
    assert indexer.indexed_documents == []


def test_catalog_lists_documents_without_processing_dependencies() -> None:
    document = IndexedDocument(
        id=UUID("12345678-1234-5678-1234-567812345678"),
        source="python-book.pdf",
        content_sha256="a" * 64,
        page_count=120,
        chunk_count=758,
    )
    repository = RecordingDocumentRepository()
    repository.documents.append(document)
    catalog = LibraryCatalog(repository)

    assert catalog.list_documents() == [document]
