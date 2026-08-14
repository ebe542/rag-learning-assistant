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


class DocumentNotFoundError(LookupError):
    """Raised when a requested library document does not exist."""


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

    def remove_document(self, document_id: UUID) -> int:
        """Remove all searchable chunks belonging to a document."""

        ...

    def replace_document(
        self,
        document: Document,
        document_id: UUID,
    ) -> list[Chunk]:
        """Replace all searchable chunks while preserving the document ID."""

        ...


class DocumentSummaryCleaner(Protocol):
    """Remove derived summaries belonging to a library document."""

    def delete_document(self, document_id: UUID) -> int:
        """Delete all persisted summary versions of a document."""
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
        summaries: DocumentSummaryCleaner | None = None,
    ) -> None:
        super().__init__(repository)
        self.extractor = extractor
        self.indexer = indexer
        self.id_factory = id_factory
        self.summaries = summaries

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

    def replace_document(
        self,
        document_id: UUID,
        path: Path,
    ) -> IndexedDocument:
        """Replace a registered document while preserving its identity."""

        existing_document = self.repository.find_by_id(document_id)

        if existing_document is None:
            raise DocumentNotFoundError(f"Document does not exist: {document_id}")

        content_sha256 = self._calculate_sha256(path)
        duplicate_document = self.repository.find_by_content_hash(content_sha256)

        if duplicate_document is not None and duplicate_document.id != document_id:
            raise DuplicateDocumentError(
                f"Document content is already indexed as {duplicate_document.source}"
            )

        document = self.extractor.extract(path)
        chunks = self.indexer.replace_document(
            document,
            document_id,
        )
        replaced_document = IndexedDocument(
            id=document_id,
            source=document.source,
            content_sha256=content_sha256,
            page_count=len(document.pages),
            chunk_count=len(chunks),
        )

        if self.summaries is not None:
            # Existing summaries describe the previous document content and must not
            # survive a successful replacement, even when the document ID stays stable.
            self.summaries.delete_document(document_id)

        self.repository.update(replaced_document)
        return replaced_document

    def remove_document(self, document_id: UUID) -> IndexedDocument:
        """Remove a document's searchable data and library metadata."""

        document = self.repository.find_by_id(document_id)

        if document is None:
            raise DocumentNotFoundError(f"Document does not exist: {document_id}")

        removed_chunk_count = self.indexer.remove_document(document_id)

        # Keep the catalog entry if vector storage does not match its metadata.
        # Removing it would hide an inconsistent index from library management.
        if removed_chunk_count != document.chunk_count:
            raise RuntimeError("Removed chunk count does not match document metadata")

        if self.summaries is not None:
            # Summaries are derived from the chunks and must not outlive their document.
            self.summaries.delete_document(document_id)

        self.repository.remove(document_id)
        return document

    @staticmethod
    def _calculate_sha256(path: Path) -> str:
        """Calculate a file hash without loading the whole PDF into memory."""

        digest = hashlib.sha256()

        with path.open("rb") as source:
            while block := source.read(1024 * 1024):
                digest.update(block)

        return digest.hexdigest()
