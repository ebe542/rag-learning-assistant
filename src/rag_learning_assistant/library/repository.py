"""Persistence interfaces and SQLite adapter for library documents."""

import sqlite3
from pathlib import Path
from typing import Protocol
from uuid import UUID

from rag_learning_assistant.library.models import IndexedDocument


class DocumentRepository(Protocol):
    """Persist and retrieve indexed document metadata."""

    def add(self, document: IndexedDocument) -> None:
        """Store one indexed document."""
        ...

    def list_all(self) -> list[IndexedDocument]:
        """Return all stored documents."""
        ...

    def find_by_content_hash(
        self,
        content_sha256: str,
    ) -> IndexedDocument | None:
        """Return the document with this content hash, if present."""
        ...

    def find_by_id(
        self,
        document_id: UUID,
    ) -> IndexedDocument | None:
        """Return the document with this ID, if present."""

        ...

    def remove(self, document_id: UUID) -> None:
        """Remove one document's metadata."""

        ...


class SqliteDocumentRepository:
    """Persist indexed document metadata in SQLite."""

    def __init__(self, database_path: str | Path) -> None:
        self.database_path = Path(database_path)
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize_database()

    def add(self, document: IndexedDocument) -> None:
        """Store one indexed document."""

        with sqlite3.connect(self.database_path) as connection:
            connection.execute(
                """
                INSERT INTO documents (
                    id,
                    source,
                    content_sha256,
                    page_count,
                    chunk_count
                )
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    str(document.id),
                    document.source,
                    document.content_sha256,
                    document.page_count,
                    document.chunk_count,
                ),
            )

    def list_all(self) -> list[IndexedDocument]:
        """Return all stored documents in insertion order."""

        with sqlite3.connect(self.database_path) as connection:
            rows = connection.execute(
                """
                SELECT
                    id,
                    source,
                    content_sha256,
                    page_count,
                    chunk_count
                FROM documents
                ORDER BY rowid
                """
            ).fetchall()

        return [
            IndexedDocument(
                id=UUID(row[0]),
                source=row[1],
                content_sha256=row[2],
                page_count=row[3],
                chunk_count=row[4],
            )
            for row in rows
        ]

    def find_by_content_hash(
        self,
        content_sha256: str,
    ) -> IndexedDocument | None:
        """Return the document with this content hash, if present."""

        with sqlite3.connect(self.database_path) as connection:
            row = connection.execute(
                """
                SELECT
                    id,
                    source,
                    content_sha256,
                    page_count,
                    chunk_count
                FROM documents
                WHERE content_sha256 = ?
                """,
                (content_sha256,),
            ).fetchone()

        if row is None:
            return None

        return self._document_from_row(row)

    def find_by_id(
        self,
        document_id: UUID,
    ) -> IndexedDocument | None:
        """Return the document with this ID, if present."""

        with sqlite3.connect(self.database_path) as connection:
            row = connection.execute(
                """
                SELECT
                    id,
                    source,
                    content_sha256,
                    page_count,
                    chunk_count
                FROM documents
                WHERE id = ?
                """,
                (str(document_id),),
            ).fetchone()

        if row is None:
            return None

        return self._document_from_row(row)

    def remove(self, document_id: UUID) -> None:
        """Remove one document's metadata."""

        with sqlite3.connect(self.database_path) as connection:
            connection.execute(
                """
                DELETE FROM documents
                WHERE id = ?
                """,
                (str(document_id),),
            )

    def _initialize_database(self) -> None:
        """Create the document metadata table when needed."""

        with sqlite3.connect(self.database_path) as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS documents (
                    id TEXT PRIMARY KEY,
                    source TEXT NOT NULL,
                    content_sha256 TEXT NOT NULL UNIQUE,
                    page_count INTEGER NOT NULL,
                    chunk_count INTEGER NOT NULL
                )
                """
            )

    @staticmethod
    def _document_from_row(
        row: tuple[str, str, str, int, int],
    ) -> IndexedDocument:
        """Convert one SQLite row into a library model."""

        return IndexedDocument(
            id=UUID(row[0]),
            source=row[1],
            content_sha256=row[2],
            page_count=row[3],
            chunk_count=row[4],
        )
