"""Persistence for user-facing learning packages."""

import sqlite3
from contextlib import closing
from pathlib import Path
from uuid import UUID

from rag_learning_assistant.learning.package_names import (
    ensure_name_reservation,
    initialize_name_registry,
    release_name_reservation,
)
from rag_learning_assistant.learning.packages import (
    LearningPackage,
    LearningPackageStatus,
)


class SqliteLearningPackageRepository:
    """Persist active learning-material selections by user-facing name."""

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
                CREATE TABLE IF NOT EXISTS learning_packages (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL UNIQUE COLLATE NOCASE,
                    document_id TEXT NOT NULL UNIQUE,
                    status TEXT NOT NULL,
                    summary_identity_fingerprint TEXT,
                    question_bank_identity_fingerprint TEXT
                )
                """
            )
            initialize_name_registry(connection)
            for row in connection.execute("SELECT id, name FROM learning_packages"):
                ensure_name_reservation(
                    connection,
                    name=row["name"],
                    owner_kind="package",
                    owner_id=UUID(row["id"]),
                )

    def save(self, package: LearningPackage) -> None:
        existing = self.find_by_id(package.id)

        if existing == package:
            return

        if existing is not None:
            raise ValueError(f"Conflicting learning package already exists: {package.id}")

        with closing(self._connect()) as connection, connection:
            ensure_name_reservation(
                connection,
                name=package.name,
                owner_kind="package",
                owner_id=package.id,
            )
            connection.execute(
                """
                INSERT INTO learning_packages (
                    id,
                    name,
                    document_id,
                    status,
                    summary_identity_fingerprint,
                    question_bank_identity_fingerprint
                )
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    str(package.id),
                    package.name,
                    str(package.document_id),
                    package.status.value,
                    package.summary_identity_fingerprint,
                    package.question_bank_identity_fingerprint,
                ),
            )

    def replace(self, package: LearningPackage) -> None:
        """Replace the current state of an existing learning package."""

        with closing(self._connect()) as connection, connection:
            existing_row = connection.execute(
                "SELECT name FROM learning_packages WHERE id = ?",
                (str(package.id),),
            ).fetchone()
            if existing_row is None:
                raise ValueError(f"Learning package does not exist: {package.id}")
            if existing_row["name"].casefold() != package.name.casefold():
                release_name_reservation(
                    connection,
                    owner_kind="package",
                    owner_id=package.id,
                )
                ensure_name_reservation(
                    connection,
                    name=package.name,
                    owner_kind="package",
                    owner_id=package.id,
                )
            cursor = connection.execute(
                """
                UPDATE learning_packages
                SET name = ?,
                    document_id = ?,
                    status = ?,
                    summary_identity_fingerprint = ?,
                    question_bank_identity_fingerprint = ?
                WHERE id = ?
                """,
                (
                    package.name,
                    str(package.document_id),
                    package.status.value,
                    package.summary_identity_fingerprint,
                    package.question_bank_identity_fingerprint,
                    str(package.id),
                ),
            )

        assert cursor.rowcount == 1

    def find_by_name(self, name: str) -> LearningPackage | None:
        with closing(self._connect()) as connection:
            row = connection.execute(
                """
                SELECT
                    id,
                    name,
                    document_id,
                    status,
                    summary_identity_fingerprint,
                    question_bank_identity_fingerprint
                FROM learning_packages
                WHERE name = ? COLLATE NOCASE
                """,
                (name,),
            ).fetchone()

        if row is None:
            return None

        return self._deserialize(row)

    def is_name_reserved(self, name: str) -> bool:
        """Report reservations across pending and materialized packages."""

        with closing(self._connect()) as connection:
            row = connection.execute(
                "SELECT 1 FROM package_names WHERE name = ? COLLATE NOCASE",
                (name,),
            ).fetchone()
        return row is not None

    def find_by_id(
        self,
        package_id: UUID,
    ) -> LearningPackage | None:
        with closing(self._connect()) as connection:
            row = connection.execute(
                """
                SELECT
                    id,
                    name,
                    document_id,
                    status,
                    summary_identity_fingerprint,
                    question_bank_identity_fingerprint
                FROM learning_packages
                WHERE id = ?
                """,
                (str(package_id),),
            ).fetchone()

        if row is None:
            return None

        return self._deserialize(row)

    def list_all(self) -> list[LearningPackage]:
        """Return all learning packages in stable user-facing order."""

        with closing(self._connect()) as connection:
            rows = connection.execute(
                """
                SELECT
                    id,
                    name,
                    document_id,
                    status,
                    summary_identity_fingerprint,
                    question_bank_identity_fingerprint
                FROM learning_packages
                ORDER BY name COLLATE NOCASE, id
                """
            ).fetchall()

        return [self._deserialize(row) for row in rows]

    def delete_document(self, document_id: UUID) -> int:
        """Delete the user-facing package belonging to one document."""

        with closing(self._connect()) as connection, connection:
            row = connection.execute(
                "SELECT id FROM learning_packages WHERE document_id = ?",
                (str(document_id),),
            ).fetchone()
            cursor = connection.execute(
                """
                DELETE FROM learning_packages
                WHERE document_id = ?
                """,
                (str(document_id),),
            )
            if row is not None:
                release_name_reservation(
                    connection,
                    owner_kind="package",
                    owner_id=UUID(row["id"]),
                )

        return cursor.rowcount

    @staticmethod
    def _deserialize(row: sqlite3.Row) -> LearningPackage:
        """Restore the validated domain model from one database row."""

        return LearningPackage(
            id=UUID(row["id"]),
            name=row["name"],
            document_id=UUID(row["document_id"]),
            status=LearningPackageStatus(row["status"]),
            summary_identity_fingerprint=row["summary_identity_fingerprint"],
            question_bank_identity_fingerprint=row["question_bank_identity_fingerprint"],
        )
