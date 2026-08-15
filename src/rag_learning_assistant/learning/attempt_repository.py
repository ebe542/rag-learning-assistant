"""Persistence for completed study-question attempts."""

import json
import sqlite3
from contextlib import closing
from datetime import datetime
from pathlib import Path
from typing import Protocol
from uuid import UUID

from rag_learning_assistant.generation import Citation
from rag_learning_assistant.learning.attempts import StudyAttempt
from rag_learning_assistant.learning.progress import (
    QuestionProgress,
    ReviewRating,
)


class StudyAttemptRepository(Protocol):
    """Append and retrieve immutable study attempts."""

    def add(self, attempt: StudyAttempt) -> None: ...

    def find_by_id(self, attempt_id: UUID) -> StudyAttempt | None: ...

    def delete_document(self, document_id: UUID) -> int: ...

    def list_question(
        self,
        document_id: UUID,
        question_bank_identity_fingerprint: str,
        question_number: int,
    ) -> list[StudyAttempt]: ...

    def record(self, attempt: StudyAttempt) -> None:
        """Atomically persist an attempt and its resulting progress."""
        ...


class SqliteStudyAttemptRepository:
    """Persist immutable study attempts in library metadata."""

    def __init__(self, database_path: Path) -> None:
        self.database_path = database_path
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize_schema()

    def add(self, attempt: StudyAttempt) -> None:
        existing = self.find_by_id(attempt.id)

        if existing == attempt:
            return

        if existing is not None:
            raise ValueError("Conflicting study attempt already exists")

        # Attempts form an append-only learning history. Existing rows must
        # never be silently replaced because that would rewrite past activity.
        with closing(self._connect()) as connection, connection:
            self._insert_attempt(connection, attempt)

    def record(self, attempt: StudyAttempt) -> None:
        """Atomically persist an attempt and its resulting progress."""

        existing = self.find_by_id(attempt.id)

        if existing is not None and existing != attempt:
            raise ValueError("Conflicting study attempt already exists")

        progress = attempt.resulting_progress

        # Progress and history describe the same completed learning event.
        # Writing both on one connection prevents partially persisted reviews.
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

            if existing is None:
                self._insert_attempt(connection, attempt)

    def find_by_id(
        self,
        attempt_id: UUID,
    ) -> StudyAttempt | None:
        with closing(self._connect()) as connection:
            row = connection.execute(
                """
                SELECT *
                FROM study_attempts
                WHERE id = ?
                """,
                (str(attempt_id),),
            ).fetchone()

        if row is None:
            return None

        return self._deserialize(row)

    def delete_document(self, document_id: UUID) -> int:
        """Delete every study attempt belonging to one document."""

        with closing(self._connect()) as connection, connection:
            cursor = connection.execute(
                """
                DELETE FROM study_attempts
                WHERE document_id = ?
                """,
                (str(document_id),),
            )

        return cursor.rowcount

    def list_question(
        self,
        document_id: UUID,
        question_bank_identity_fingerprint: str,
        question_number: int,
    ) -> list[StudyAttempt]:
        """Return one question's attempts from oldest to newest."""

        with closing(self._connect()) as connection:
            rows = connection.execute(
                """
                SELECT *
                FROM study_attempts
                WHERE document_id = ?
                  AND question_bank_identity_fingerprint = ?
                  AND question_number = ?
                ORDER BY answered_at, id
                """,
                (
                    str(document_id),
                    question_bank_identity_fingerprint,
                    question_number,
                ),
            ).fetchall()

        return [self._deserialize(row) for row in rows]

    @staticmethod
    def _insert_attempt(
        connection: sqlite3.Connection,
        attempt: StudyAttempt,
    ) -> None:
        citations_json = json.dumps(
            [
                {
                    "number": citation.number,
                    "source": citation.source,
                    "page_number": citation.page_number,
                    "chunk_index": citation.chunk_index,
                    "excerpt": citation.excerpt,
                }
                for citation in attempt.citations
            ],
            ensure_ascii=False,
        )
        progress = attempt.resulting_progress

        connection.execute(
            """
            INSERT INTO study_attempts (
                id,
                document_id,
                question_bank_identity_fingerprint,
                question_number,
                question_text,
                answer_text,
                expected_answer,
                citations_json,
                rating,
                answered_at,
                repetition_count,
                interval_days,
                ease_factor,
                due_at,
                last_reviewed_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                str(attempt.id),
                str(attempt.document_id),
                attempt.question_bank_identity_fingerprint,
                attempt.question_number,
                attempt.question_text,
                attempt.answer_text,
                attempt.expected_answer,
                citations_json,
                attempt.rating.value,
                attempt.answered_at.isoformat(),
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

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.database_path)
        connection.row_factory = sqlite3.Row
        return connection

    def _initialize_schema(self) -> None:
        with closing(self._connect()) as connection, connection:
            # Session recording owns the transaction for both current progress
            # and immutable history, so it must initialize both schemas.
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
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS study_attempts (
                    id TEXT PRIMARY KEY,
                    document_id TEXT NOT NULL,
                    question_bank_identity_fingerprint TEXT NOT NULL,
                    question_number INTEGER NOT NULL,
                    question_text TEXT NOT NULL,
                    answer_text TEXT NOT NULL,
                    expected_answer TEXT NOT NULL,
                    citations_json TEXT NOT NULL,
                    rating TEXT NOT NULL,
                    answered_at TEXT NOT NULL,
                    repetition_count INTEGER NOT NULL,
                    interval_days INTEGER NOT NULL,
                    ease_factor REAL NOT NULL,
                    due_at TEXT NOT NULL,
                    last_reviewed_at TEXT
                )
                """
            )

    @staticmethod
    def _deserialize(row: sqlite3.Row) -> StudyAttempt:
        """Reconstruct one validated domain object from persistent data."""

        document_id = UUID(row["document_id"])
        identity_fingerprint = row["question_bank_identity_fingerprint"]
        question_number = row["question_number"]
        answered_at = datetime.fromisoformat(row["answered_at"])

        citations = tuple(
            Citation(
                number=item["number"],
                source=item["source"],
                page_number=item["page_number"],
                chunk_index=item["chunk_index"],
                excerpt=item["excerpt"],
            )
            for item in json.loads(row["citations_json"])
        )
        progress = QuestionProgress(
            document_id=document_id,
            question_bank_identity_fingerprint=identity_fingerprint,
            question_number=question_number,
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

        return StudyAttempt(
            id=UUID(row["id"]),
            document_id=document_id,
            question_bank_identity_fingerprint=identity_fingerprint,
            question_number=question_number,
            question_text=row["question_text"],
            answer_text=row["answer_text"],
            expected_answer=row["expected_answer"],
            citations=citations,
            rating=ReviewRating(row["rating"]),
            answered_at=answered_at,
            resulting_progress=progress,
        )
