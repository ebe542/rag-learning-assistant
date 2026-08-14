from typing import Protocol
from uuid import UUID

from rag_learning_assistant.application.library import DocumentNotFoundError
from rag_learning_assistant.generation import PersistedDocumentSummary
from rag_learning_assistant.library import DocumentRepository


class DocumentSummaryNotFoundError(LookupError):
    """Raised when a requested persisted summary identity does not exist."""


class DocumentSummaryReader(Protocol):
    """Read persisted final summaries without exposing write operations."""

    def list_document(
        self,
        document_id: UUID,
    ) -> list[PersistedDocumentSummary]:
        """Return all stored summary versions for one document."""
        ...

    def find(
        self,
        document_id: UUID,
        identity_fingerprint: str,
    ) -> PersistedDocumentSummary | None:
        """Return one exact persisted summary identity."""
        ...


class DocumentSummaryCatalog:
    """Provide read-only access to summaries of registered documents."""

    def __init__(
        self,
        documents: DocumentRepository,
        summaries: DocumentSummaryReader,
    ) -> None:
        self.documents = documents
        self.summaries = summaries

    def list_document_summaries(
        self,
        document_id: UUID,
    ) -> list[PersistedDocumentSummary]:
        """Return stored summaries after validating library membership."""

        if self.documents.find_by_id(document_id) is None:
            raise DocumentNotFoundError(
                f"Document does not exist: {document_id}",
            )

        return self.summaries.list_document(document_id)

    def get_document_summary(
        self,
        document_id: UUID,
        identity_fingerprint: str,
    ) -> PersistedDocumentSummary:
        """Return one exact persisted summary after validating its document."""

        if self.documents.find_by_id(document_id) is None:
            raise DocumentNotFoundError(
                f"Document does not exist: {document_id}",
            )

        summary = self.summaries.find(
            document_id,
            identity_fingerprint,
        )

        if summary is None:
            raise DocumentSummaryNotFoundError(
                f"Stored document summary does not exist: {document_id}/{identity_fingerprint}"
            )

        return summary
