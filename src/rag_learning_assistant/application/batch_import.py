"""Sequential batch import for document libraries."""

from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Protocol

from rag_learning_assistant.application.library import (
    DuplicateDocumentError,
)
from rag_learning_assistant.library import IndexedDocument


class ImportStatus(StrEnum):
    """Possible outcomes for one imported path."""

    ADDED = "added"
    SKIPPED = "skipped"
    FAILED = "failed"


@dataclass(frozen=True, slots=True)
class ImportOutcome:
    """Result of processing one input path."""

    path: Path
    status: ImportStatus
    document: IndexedDocument | None = None
    message: str | None = None


class DocumentLibrary(Protocol):
    """Add documents to a persistent library."""

    def add_document(self, path: Path) -> IndexedDocument:
        """Add one document and return its metadata."""
        ...


class BatchImportService:
    """Add multiple documents sequentially to one library."""

    def __init__(self, library: DocumentLibrary) -> None:
        self.library = library

    def add_documents(
        self,
        paths: Sequence[Path],
    ) -> list[ImportOutcome]:
        """Add every input document in its original order."""

        outcomes: list[ImportOutcome] = []

        for path in paths:
            try:
                document = self.library.add_document(path)
            except DuplicateDocumentError as exc:
                outcomes.append(
                    ImportOutcome(
                        path=path,
                        status=ImportStatus.SKIPPED,
                        message=str(exc),
                    )
                )
            except Exception as exc:
                # One invalid file must not abort the remaining batch.
                outcomes.append(
                    ImportOutcome(
                        path=path,
                        status=ImportStatus.FAILED,
                        message=str(exc),
                    )
                )
            else:
                outcomes.append(
                    ImportOutcome(
                        path=path,
                        status=ImportStatus.ADDED,
                        document=document,
                    )
                )

        return outcomes
