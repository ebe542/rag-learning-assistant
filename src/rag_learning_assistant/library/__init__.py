"""Persistent document-library management."""

from rag_learning_assistant.library.languages import DocumentLanguage, detect_document_language
from rag_learning_assistant.library.models import IndexedDocument
from rag_learning_assistant.library.repository import (
    DocumentRepository,
    SqliteDocumentRepository,
)

__all__ = [
    "DocumentRepository",
    "IndexedDocument",
    "DocumentLanguage",
    "detect_document_language",
    "SqliteDocumentRepository",
]
