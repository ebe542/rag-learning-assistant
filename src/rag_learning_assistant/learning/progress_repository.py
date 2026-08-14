"""Persistence for study-question review progress."""

import sqlite3
from contextlib import closing
from datetime import datetime
from pathlib import Path
from typing import Protocol
from uuid import UUID

from rag_learning_assistant.learning.progress import QuestionProgress


class QuestionProgressRepository(Protocol):
    """Persist and retrieve the current schedule of study questions."""

    def find(
        self,
        document_id: UUID,
        question_bank_identity_fingerprint: str,
        question_number: int,
    ) -> QuestionProgress | None: ...

    def save(self, progress: QuestionProgress) -> None: ...


class SqliteQuestionProgressRepository:
    """Persist current question schedules in library metadata."""

    def __init__(self, database_path: Path) -> None:
        self.database_path = database_path
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize_schema()

    def find(
        self,
        document_id: UUID,
        question_bank_identity_fingerprint: str,
        question_number: int,
    ) -> QuestionProgress | None:
        with closing(self._connect()) as connection:
            row = connection.execute(
                """
                SELECT
                    document_id,
                    question_bank_identity_fingerprint,
                    question_number,
                    repetition_count,
                    interval_days,
                    ease_factor,
                    due_at,
                    last_reviewed_at
                FROM question_progress
                WHERE document_id = ?
                AND question_bank_identity_fingerprint = ?
                AND question_number = ?
                """,
                (
                    str(document_id),
                    question_bank_identity_fingerprint,
                    question_number,
                ),
            ).fetchone()

        if row is None:
            return None

        return QuestionProgress(
            document_id=UUID(row["document_id"]),
            question_bank_identity_fingerprint=row["question_bank_identity_fingerprint"],
            question_number=row["question_number"],
            repetition_count=row["repetition_count"],
            interval_days=row["interval_days"],
            ease_factor=row["ease_factor"],
            due_at=datetime.fromisoformat(row["due_at"]),
            last_reviewed_at=(
                datetime.fromisoformat(row["last_reviewed_at"])
                if row["last_reviewed_at"] is not None
                else None
            ),
        )

    def save(self, progress: QuestionProgress) -> None:
        # Review progress represents current state, so saving the same question
        # intentionally replaces its previous schedule.
        with closing(self._connect()) as connection, connection:
            connection.execute(
                """
                INSERT INTO question_progress (
                    document_id,
                    question_bank_identity_fingerprint,
                    question_number,
                    repetition_count,
                    interval_days,
                    ease_factor,
                    due_at,
                    last_reviewed_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT (
                    document_id,
                    question_bank_identity_fingerprint,
                    question_number
                )
                DO UPDATE SET
                    repetition_count = excluded.repetition_count,
                    interval_days = excluded.interval_days,
                    ease_factor = excluded.ease_factor,
                    due_at = excluded.due_at,
                    last_reviewed_at = excluded.last_reviewed_at
                """,
                (
                    str(progress.document_id),
                    progress.question_bank_identity_fingerprint,
                    progress.question_number,
                    progress.repetition_count,
                    progress.interval_days,
                    progress.ease_factor,
                    progress.due_at.isoformat(),
                    (
                        progress.last_reviewed_at.isoformat()
                        if progress.last_reviewed_at is not None
                        else None
                    ),
                ),
            )

    def delete_document(self, document_id: UUID) -> int:
        """Delete all review progress belonging to one library document."""

        with closing(self._connect()) as connection, connection:
            cursor = connection.execute(
                """
                DELETE FROM question_progress
                WHERE document_id = ?
                """,
                (str(document_id),),
            )

        return cursor.rowcount

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.database_path)
        connection.row_factory = sqlite3.Row
        return connection

    def _initialize_schema(self) -> None:
        # A transaction context commits or rolls back, but closing remains an
        # explicit responsibility of the sqlite3 caller.
        with closing(self._connect()) as connection, connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS question_progress (
                    document_id TEXT NOT NULL,
                    question_bank_identity_fingerprint TEXT NOT NULL,
                    question_number INTEGER NOT NULL,
                    repetition_count INTEGER NOT NULL,
                    interval_days INTEGER NOT NULL,
                    ease_factor REAL NOT NULL,
                    due_at TEXT NOT NULL,
                    last_reviewed_at TEXT,
                    PRIMARY KEY (
                        document_id,
                        question_bank_identity_fingerprint,
                        question_number
                    )
                )
                """
            )
