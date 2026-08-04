"""Application services coordinating the processing pipeline."""

from rag_learning_assistant.application.document_search import (
    DocumentSearchService,
)
from rag_learning_assistant.application.library import (
    DuplicateDocumentError,
    LibraryCatalog,
    LibraryService,
)

__all__ = [
    "DocumentSearchService",
    "DuplicateDocumentError",
    "LibraryCatalog",
    "LibraryService",
]
