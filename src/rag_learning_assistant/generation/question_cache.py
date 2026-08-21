"""Persistent intermediate results for resumable question generation."""

import json
import sqlite3
from contextlib import closing
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from rag_learning_assistant.generation.prompts import PromptReference
from rag_learning_assistant.generation.question_parser import (
    GeneratedQuestionDraft,
    QuestionGenerationResult,
)


@dataclass(frozen=True, slots=True)
class CachedQuestionBatch:
    """Store one validated question-generation batch for later resumption."""

    identity_fingerprint: str
    batch_number: int
    first_question_number: int
    last_question_number: int
    result: QuestionGenerationResult

    def __post_init__(self) -> None:
        is_lowercase_sha256 = len(self.identity_fingerprint) == 64 and all(
            character in "0123456789abcdef" for character in self.identity_fingerprint
        )
        if not is_lowercase_sha256:
            raise ValueError("Question batch identity must be a lowercase SHA-256 hex digest")

        if self.batch_number < 1:
            raise ValueError("Question batch number must be positive")

        if self.first_question_number < 1:
            raise ValueError("First question number must be positive")

        if self.last_question_number < 1:
            raise ValueError("Last question number must be positive")

        if self.last_question_number < self.first_question_number:
            raise ValueError("Last question number must not precede first question number")
        expected_question_numbers = tuple(
            range(
                self.first_question_number,
                self.last_question_number + 1,
            )
        )
        actual_question_numbers = tuple(question.number for question in self.result.questions)

        if actual_question_numbers != expected_question_numbers:
            raise ValueError("Generated question numbers must match the cached batch range")


class QuestionBatchCache(Protocol):
    """Load and persist successful question-generation batches."""

    def find_batch(
        self,
        identity_fingerprint: str,
        batch_number: int,
    ) -> CachedQuestionBatch | None:
        """Return one cached batch or None."""

        ...

    def save_batch(self, batch: CachedQuestionBatch) -> None:
        """Persist one completed batch idempotently."""

        ...


class SqliteQuestionBatchCache:
    """Persist validated question batches for resumable generation."""

    def __init__(self, database_path: str | Path) -> None:
        self.database_path = Path(database_path)
        self.database_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )
        self._initialize_database()

    def save_batch(self, batch: CachedQuestionBatch) -> None:
        """Persist one completed batch idempotently."""

        questions = json.dumps(
            [
                {
                    "number": question.number,
                    "text": question.text,
                    "expected_answer": question.expected_answer,
                    "citation_numbers": question.citation_numbers,
                }
                for question in batch.result.questions
            ],
            ensure_ascii=False,
            separators=(",", ":"),
        )
        prompt_references = json.dumps(
            [
                {
                    "name": reference.name,
                    "version": reference.version,
                    "fingerprint": reference.fingerprint,
                }
                for reference in batch.result.prompt_references
            ],
            separators=(",", ":"),
        )

        with closing(sqlite3.connect(self.database_path)) as connection, connection:
            connection.execute(
                """
                INSERT INTO question_batches (
                    identity_fingerprint,
                    batch_number,
                    first_question_number,
                    last_question_number,
                    questions,
                    prompt_references
                )
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT (
                    identity_fingerprint,
                    batch_number
                )
                DO NOTHING
                """,
                (
                    batch.identity_fingerprint,
                    batch.batch_number,
                    batch.first_question_number,
                    batch.last_question_number,
                    questions,
                    prompt_references,
                ),
            )

        stored = self.find_batch(
            identity_fingerprint=batch.identity_fingerprint,
            batch_number=batch.batch_number,
        )

        if stored is None:
            raise RuntimeError("Cached question batch was not persisted")

        # A resumed run may save the same completed batch again. Different
        # content for the same identity and batch number indicates corruption.
        if stored != batch:
            raise ValueError("Cached question batch conflicts with existing data")

    def find_batch(
        self,
        identity_fingerprint: str,
        batch_number: int,
    ) -> CachedQuestionBatch | None:
        """Return one cached question batch or None."""

        with closing(sqlite3.connect(self.database_path)) as connection:
            row = connection.execute(
                """
                SELECT
                    first_question_number,
                    last_question_number,
                    questions,
                    prompt_references
                FROM question_batches
                WHERE identity_fingerprint = ?
                  AND batch_number = ?
                """,
                (
                    identity_fingerprint,
                    batch_number,
                ),
            ).fetchone()

        if row is None:
            return None

        (
            first_question_number,
            last_question_number,
            questions_json,
            prompt_references_json,
        ) = row

        questions = tuple(
            GeneratedQuestionDraft(
                number=value["number"],
                text=value["text"],
                expected_answer=value["expected_answer"],
                citation_numbers=tuple(value["citation_numbers"]),
            )
            for value in json.loads(questions_json)
        )
        prompt_references = tuple(
            PromptReference(
                name=value["name"],
                version=value["version"],
                fingerprint=value["fingerprint"],
            )
            for value in json.loads(prompt_references_json)
        )

        return CachedQuestionBatch(
            identity_fingerprint=identity_fingerprint,
            batch_number=batch_number,
            first_question_number=first_question_number,
            last_question_number=last_question_number,
            result=QuestionGenerationResult(
                questions=questions,
                prompt_references=prompt_references,
            ),
        )

    def _initialize_database(self) -> None:
        """Create the cache table without modifying existing library data."""

        with closing(sqlite3.connect(self.database_path)) as connection, connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS question_batches (
                    identity_fingerprint TEXT NOT NULL,
                    batch_number INTEGER NOT NULL,
                    first_question_number INTEGER NOT NULL,
                    last_question_number INTEGER NOT NULL,
                    questions TEXT NOT NULL,
                    prompt_references TEXT NOT NULL,
                    PRIMARY KEY (
                        identity_fingerprint,
                        batch_number
                    )
                )
                """
            )
