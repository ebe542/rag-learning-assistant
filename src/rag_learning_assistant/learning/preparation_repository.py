"""SQLite persistence for pending package preparations."""

import sqlite3
from contextlib import closing
from datetime import UTC, datetime, timedelta
from pathlib import Path
from uuid import UUID

from rag_learning_assistant.learning.package_names import (
    ensure_name_reservation,
    initialize_name_registry,
)
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
                    status TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    lease_token TEXT,
                    lease_expires_at TEXT,
                    failure_message TEXT
                )
                """
            )
            columns = {
                row["name"] for row in connection.execute("PRAGMA table_info(package_preparations)")
            }
            additions = {
                "created_at": "TEXT",
                "updated_at": "TEXT",
                "lease_token": "TEXT",
                "lease_expires_at": "TEXT",
                "failure_message": "TEXT",
            }
            for name, data_type in additions.items():
                if name not in columns:
                    connection.execute(
                        f"ALTER TABLE package_preparations ADD COLUMN {name} {data_type}"
                    )
            initialized_at = _serialize_datetime(datetime.now(UTC))
            connection.execute(
                "UPDATE package_preparations SET created_at = ? WHERE created_at IS NULL",
                (initialized_at,),
            )
            connection.execute(
                "UPDATE package_preparations SET updated_at = created_at WHERE updated_at IS NULL"
            )
            initialize_name_registry(connection)
            for row in connection.execute("SELECT id, name FROM package_preparations"):
                ensure_name_reservation(
                    connection,
                    name=row["name"],
                    owner_kind="preparation",
                    owner_id=UUID(row["id"]),
                )

    def save(self, preparation: PackagePreparation) -> None:
        with closing(self._connect()) as connection, connection:
            ensure_name_reservation(
                connection,
                name=preparation.name,
                owner_kind="preparation",
                owner_id=preparation.id,
            )
            connection.execute(
                """
                INSERT INTO package_preparations (
                    id, name, source_filename, stored_filename,
                    question_count, size_bytes, status, created_at, updated_at,
                    lease_token, lease_expires_at, failure_message
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    str(preparation.id),
                    preparation.name,
                    preparation.source_filename,
                    preparation.stored_filename,
                    preparation.question_count,
                    preparation.size_bytes,
                    preparation.status.value,
                    _serialize_datetime(preparation.created_at),
                    _serialize_datetime(preparation.updated_at),
                    str(preparation.lease_token) if preparation.lease_token is not None else None,
                    (
                        _serialize_datetime(preparation.lease_expires_at)
                        if preparation.lease_expires_at is not None
                        else None
                    ),
                    preparation.failure_message,
                ),
            )

    def find_by_name(self, name: str) -> PackagePreparation | None:
        with closing(self._connect()) as connection:
            row = connection.execute(
                """
                SELECT id, name, source_filename, stored_filename,
                       question_count, size_bytes, status, created_at, updated_at,
                       lease_token, lease_expires_at, failure_message
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
                       question_count, size_bytes, status, created_at, updated_at,
                       lease_token, lease_expires_at, failure_message
                FROM package_preparations
                ORDER BY name COLLATE NOCASE, id
                """
            ).fetchall()
        return [self._deserialize(row) for row in rows]

    def claim_next(
        self,
        *,
        lease_token: UUID,
        now: datetime,
        lease_duration: timedelta,
    ) -> PackagePreparation | None:
        """Atomically claim the oldest pending or abandoned active request."""

        _validate_lease_arguments(now, lease_duration)
        serialized_now = _serialize_datetime(now)
        with closing(self._connect()) as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                """
                SELECT id, status
                FROM package_preparations
                WHERE status = ?
                   OR (
                       status IN (?, ?, ?)
                       AND lease_expires_at <= ?
                   )
                ORDER BY
                    CASE WHEN status = ? THEN 0 ELSE 1 END,
                    created_at,
                    id
                LIMIT 1
                """,
                (
                    PackagePreparationStatus.PENDING.value,
                    PackagePreparationStatus.INDEXING.value,
                    PackagePreparationStatus.SUMMARIZING.value,
                    PackagePreparationStatus.GENERATING_QUESTIONS.value,
                    serialized_now,
                    PackagePreparationStatus.PENDING.value,
                ),
            ).fetchone()
            if row is None:
                connection.commit()
                return None
            status = PackagePreparationStatus(row["status"])
            if status is PackagePreparationStatus.PENDING:
                status = PackagePreparationStatus.INDEXING
            connection.execute(
                """
                UPDATE package_preparations
                SET status = ?, updated_at = ?, lease_token = ?, lease_expires_at = ?
                WHERE id = ?
                """,
                (
                    status.value,
                    serialized_now,
                    str(lease_token),
                    _serialize_datetime(now + lease_duration),
                    row["id"],
                ),
            )
            claimed = self._find_by_id(connection, UUID(row["id"]))
            connection.commit()
        assert claimed is not None
        return claimed

    def advance(
        self,
        preparation_id: UUID,
        *,
        lease_token: UUID,
        current_status: PackagePreparationStatus,
        next_status: PackagePreparationStatus,
        now: datetime,
        lease_duration: timedelta,
    ) -> PackagePreparation:
        """Advance one leased request by exactly one allowed phase."""

        allowed_next = {
            PackagePreparationStatus.INDEXING: PackagePreparationStatus.SUMMARIZING,
            PackagePreparationStatus.SUMMARIZING: (PackagePreparationStatus.GENERATING_QUESTIONS),
        }
        if allowed_next.get(current_status) is not next_status:
            raise ValueError(f"Invalid preparation transition: {current_status} -> {next_status}")
        _validate_lease_arguments(now, lease_duration)
        with closing(self._connect()) as connection, connection:
            cursor = connection.execute(
                """
                UPDATE package_preparations
                SET status = ?, updated_at = ?, lease_expires_at = ?
                WHERE id = ? AND status = ? AND lease_token = ? AND lease_expires_at > ?
                """,
                (
                    next_status.value,
                    _serialize_datetime(now),
                    _serialize_datetime(now + lease_duration),
                    str(preparation_id),
                    current_status.value,
                    str(lease_token),
                    _serialize_datetime(now),
                ),
            )
            if cursor.rowcount != 1:
                raise ValueError("Package preparation lease is no longer valid")
            advanced = self._find_by_id(connection, preparation_id)
        assert advanced is not None
        return advanced

    def mark_failed(
        self,
        preparation_id: UUID,
        *,
        lease_token: UUID,
        now: datetime,
        message: str,
    ) -> PackagePreparation:
        """Record a leased phase failure and release the worker lease."""

        if not message.strip():
            raise ValueError("Failure message must not be blank")
        with closing(self._connect()) as connection, connection:
            cursor = connection.execute(
                """
                UPDATE package_preparations
                SET status = ?, updated_at = ?, lease_token = NULL,
                    lease_expires_at = NULL, failure_message = ?
                WHERE id = ? AND lease_token = ? AND lease_expires_at > ?
                """,
                (
                    PackagePreparationStatus.FAILED.value,
                    _serialize_datetime(now),
                    message,
                    str(preparation_id),
                    str(lease_token),
                    _serialize_datetime(now),
                ),
            )
            if cursor.rowcount != 1:
                raise ValueError("Package preparation lease is no longer valid")
            failed = self._find_by_id(connection, preparation_id)
        assert failed is not None
        return failed

    def retry_failed(self, preparation_id: UUID, *, now: datetime) -> PackagePreparation:
        """Return a failed request to the queue without losing its upload."""

        with closing(self._connect()) as connection, connection:
            cursor = connection.execute(
                """
                UPDATE package_preparations
                SET status = ?, updated_at = ?, failure_message = NULL
                WHERE id = ? AND status = ?
                """,
                (
                    PackagePreparationStatus.PENDING.value,
                    _serialize_datetime(now),
                    str(preparation_id),
                    PackagePreparationStatus.FAILED.value,
                ),
            )
            if cursor.rowcount != 1:
                raise ValueError("Only failed package preparations can be retried")
            retried = self._find_by_id(connection, preparation_id)
        assert retried is not None
        return retried

    @staticmethod
    def _find_by_id(
        connection: sqlite3.Connection,
        preparation_id: UUID,
    ) -> PackagePreparation | None:
        row = connection.execute(
            """
            SELECT id, name, source_filename, stored_filename,
                   question_count, size_bytes, status, created_at, updated_at,
                   lease_token, lease_expires_at, failure_message
            FROM package_preparations
            WHERE id = ?
            """,
            (str(preparation_id),),
        ).fetchone()
        return SqlitePackagePreparationRepository._deserialize(row) if row is not None else None

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
            created_at=_deserialize_datetime(row["created_at"]),
            updated_at=_deserialize_datetime(row["updated_at"]),
            lease_token=UUID(row["lease_token"]) if row["lease_token"] is not None else None,
            lease_expires_at=(
                _deserialize_datetime(row["lease_expires_at"])
                if row["lease_expires_at"] is not None
                else None
            ),
            failure_message=row["failure_message"],
        )


def _serialize_datetime(value: datetime) -> str:
    if value.tzinfo is None:
        raise ValueError("Package preparation timestamp must be timezone-aware")
    return value.astimezone(UTC).isoformat()


def _deserialize_datetime(value: str) -> datetime:
    return datetime.fromisoformat(value)


def _validate_lease_arguments(now: datetime, lease_duration: timedelta) -> None:
    if now.tzinfo is None:
        raise ValueError("Lease timestamp must be timezone-aware")
    if lease_duration <= timedelta(0):
        raise ValueError("Lease duration must be positive")
