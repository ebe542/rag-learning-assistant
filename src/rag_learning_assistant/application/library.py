"""Application service for adding documents to a library."""

import hashlib
from collections.abc import Callable
from pathlib import Path
from typing import Protocol
from uuid import UUID, uuid4

from rag_learning_assistant.chunking import Chunk
from rag_learning_assistant.ingestion import Document
from rag_learning_assistant.library import (
    DocumentRepository,
    IndexedDocument,
)


class DuplicateDocumentError(ValueError):
    """Raised when identical document content is already registered."""


class DocumentExtractor(Protocol):
    """Extract a document from a source file."""

    def extract(self, path: Path) -> Document:
        """Extract the document at the given path."""
        ...


class DocumentIndexer(Protocol):
    """Index all searchable chunks of a document."""

    def index_document(
        self,
        document: Document,
        *,
        document_id: UUID | None = None,
    ) -> list[Chunk]:
        """Index a document and return its chunks."""
        ...


class LibraryCatalog:
    """Provide read-only access to registered library documents."""

    def __init__(self, repository: DocumentRepository) -> None:
        self.repository = repository

    def list_documents(self) -> list[IndexedDocument]:
        """Return all documents registered in the library."""

        return self.repository.list_all()


class LibraryService(LibraryCatalog):
    """Coordinate duplicate detection, indexing, and registration."""

    def __init__(
        self,
        repository: DocumentRepository,
        extractor: DocumentExtractor,
        indexer: DocumentIndexer,
        id_factory: Callable[[], UUID] = uuid4,
    ) -> None:
        super().__init__(repository)
        self.extractor = extractor
        self.indexer = indexer
        self.id_factory = id_factory

    def add_document(self, path: Path) -> IndexedDocument:
        """Index a file and register its persistent library metadata."""

        content_sha256 = self._calculate_sha256(path)
        existing_document = self.repository.find_by_content_hash(content_sha256)

        if existing_document is not None:
            raise DuplicateDocumentError(
                f"Document content is already indexed as {existing_document.source}"
            )

        document_id = self.id_factory()
        document = self.extractor.extract(path)
        chunks = self.indexer.index_document(
            document,
            document_id=document_id,
        )
        indexed_document = IndexedDocument(
            id=document_id,
            source=document.source,
            content_sha256=content_sha256,
            page_count=len(document.pages),
            chunk_count=len(chunks),
        )
        self.repository.add(indexed_document)
        return indexed_document

    @staticmethod
    def _calculate_sha256(path: Path) -> str:
        """Calculate a file hash without loading the whole PDF into memory."""

        digest = hashlib.sha256()

        with path.open("rb") as source:
            while block := source.read(1024 * 1024):
                digest.update(block)

        return digest.hexdigest()
