"""SQLite persistence for pending package preparations."""

import sqlite3
from contextlib import closing
from pathlib import Path
from uuid import UUID

from rag_learning_assistant.learning.preparations import (
    PackagePreparation,
    PackagePreparationStatus,
)


class SqlitePackagePreparationRepository:
    """Persist uploaded PDFs waiting for the preparation pipeline."""

    def __init__(self, database_path: Path) -> None:
        self.database_path = database_path
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize_schema()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.database_path)
        connection.row_factory = sqlite3.Row
        return connection

    def _initialize_schema(self) -> None:
        with closing(self._connect()) as connection, connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS package_preparations (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL UNIQUE COLLATE NOCASE,
                    source_filename TEXT NOT NULL,
                    stored_filename TEXT NOT NULL UNIQUE,
                    question_count INTEGER NOT NULL,
                    size_bytes INTEGER NOT NULL,
                    status TEXT NOT NULL
                )
                """
            )

    def save(self, preparation: PackagePreparation) -> None:
        with closing(self._connect()) as connection, connection:
            connection.execute(
                """
                INSERT INTO package_preparations (
                    id, name, source_filename, stored_filename,
                    question_count, size_bytes, status
                )
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    str(preparation.id),
                    preparation.name,
                    preparation.source_filename,
                    preparation.stored_filename,
                    preparation.question_count,
                    preparation.size_bytes,
                    preparation.status.value,
                ),
            )

    def find_by_name(self, name: str) -> PackagePreparation | None:
        with closing(self._connect()) as connection:
            row = connection.execute(
                """
                SELECT id, name, source_filename, stored_filename,
                       question_count, size_bytes, status
                FROM package_preparations
                WHERE name = ? COLLATE NOCASE
                """,
                (name,),
            ).fetchone()
        return self._deserialize(row) if row is not None else None

    def list_all(self) -> list[PackagePreparation]:
        with closing(self._connect()) as connection:
            rows = connection.execute(
                """
                SELECT id, name, source_filename, stored_filename,
                       question_count, size_bytes, status
                FROM package_preparations
                ORDER BY name COLLATE NOCASE, id
                """
            ).fetchall()
        return [self._deserialize(row) for row in rows]

    @staticmethod
    def _deserialize(row: sqlite3.Row) -> PackagePreparation:
        return PackagePreparation(
            id=UUID(row["id"]),
            name=row["name"],
            source_filename=row["source_filename"],
            stored_filename=row["stored_filename"],
            question_count=row["question_count"],
            size_bytes=row["size_bytes"],
            status=PackagePreparationStatus(row["status"]),
        )
