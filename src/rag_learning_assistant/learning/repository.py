import json
import sqlite3
from contextlib import closing
from pathlib import Path
from typing import Protocol
from uuid import UUID

from rag_learning_assistant.generation import Citation, PromptReference
from rag_learning_assistant.learning.models import (
    QuestionBank,
    StudyQuestion,
)


class QuestionBankRepository(Protocol):
    """Persist and retrieve exact generated question-bank identities."""

    def find(
        self,
        document_id: UUID,
        identity_fingerprint: str,
    ) -> QuestionBank | None: ...

    def save(self, bank: QuestionBank) -> None: ...

    def replace(self, bank: QuestionBank) -> None: ...


class SqliteQuestionBankRepository:
    """Persist complete grounded question banks in library metadata."""

    def __init__(self, database_path: Path) -> None:
        self.database_path = database_path
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize_schema()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.database_path)
        connection.row_factory = sqlite3.Row
        return connection

    def _initialize_schema(self) -> None:
        # sqlite3 manages the transaction context but does not close the
        # connection itself.
        with closing(self._connect()) as connection, connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS question_banks (
                    document_id TEXT NOT NULL,
                    identity_fingerprint TEXT NOT NULL,
                    source TEXT NOT NULL,
                    questions_json TEXT NOT NULL,
                    prompt_references_json TEXT NOT NULL,
                    PRIMARY KEY (document_id, identity_fingerprint)
                )
                """
            )

    def save(self, bank: QuestionBank) -> None:
        existing = self.find(
            bank.document_id,
            bank.identity_fingerprint,
        )

        if existing == bank:
            return

        if existing is not None:
            raise ValueError(
                "Conflicting question bank already exists",
            )

        questions_json, prompts_json = self._serialize(bank)

        with closing(self._connect()) as connection, connection:
            connection.execute(
                """
                INSERT INTO question_banks (
                    document_id,
                    identity_fingerprint,
                    source,
                    questions_json,
                    prompt_references_json
                )
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    str(bank.document_id),
                    bank.identity_fingerprint,
                    bank.source,
                    questions_json,
                    prompts_json,
                ),
            )

    def replace(self, bank: QuestionBank) -> None:
        """Explicitly replace a bank for the same generation identity."""

        questions_json, prompts_json = self._serialize(bank)

        with closing(self._connect()) as connection, connection:
            cursor = connection.execute(
                """
                UPDATE question_banks
                SET source = ?,
                    questions_json = ?,
                    prompt_references_json = ?
                WHERE document_id = ?
                AND identity_fingerprint = ?
                """,
                (
                    bank.source,
                    questions_json,
                    prompts_json,
                    str(bank.document_id),
                    bank.identity_fingerprint,
                ),
            )

        if cursor.rowcount == 0:
            # Forced generation may also be the first completed run for this
            # exact identity.
            self.save(bank)

    def list_document(
        self,
        document_id: UUID,
    ) -> list[QuestionBank]:
        """Return every persisted question-bank identity for one document."""

        with closing(self._connect()) as connection:
            rows = connection.execute(
                """
                SELECT
                    identity_fingerprint,
                    source,
                    questions_json,
                    prompt_references_json
                FROM question_banks
                WHERE document_id = ?
                ORDER BY identity_fingerprint
                """,
                (str(document_id),),
            ).fetchall()

        return [
            self._deserialize(
                document_id=document_id,
                identity_fingerprint=row["identity_fingerprint"],
                row=row,
            )
            for row in rows
        ]

    def delete_document(self, document_id: UUID) -> int:
        """Delete every persisted question bank belonging to one document."""

        with closing(self._connect()) as connection, connection:
            cursor = connection.execute(
                """
                DELETE FROM question_banks
                WHERE document_id = ?
                """,
                (str(document_id),),
            )

            return cursor.rowcount

    def find(
        self,
        document_id: UUID,
        identity_fingerprint: str,
    ) -> QuestionBank | None:
        with closing(self._connect()) as connection:
            row = connection.execute(
                """
                SELECT
                    source,
                    questions_json,
                    prompt_references_json
                FROM question_banks
                WHERE document_id = ?
                  AND identity_fingerprint = ?
                """,
                (
                    str(document_id),
                    identity_fingerprint,
                ),
            ).fetchone()

        if row is None:
            return None

        return self._deserialize(
            document_id=document_id,
            identity_fingerprint=identity_fingerprint,
            row=row,
        )

    @staticmethod
    def _serialize(bank: QuestionBank) -> tuple[str, str]:
        questions = [
            {
                "number": question.number,
                "text": question.text,
                "expected_answer": question.expected_answer,
                "citations": [
                    {
                        "number": citation.number,
                        "source": citation.source,
                        "page_number": citation.page_number,
                        "chunk_index": citation.chunk_index,
                        "excerpt": citation.excerpt,
                    }
                    for citation in question.citations
                ],
            }
            for question in bank.questions
        ]
        prompts = [
            {
                "name": reference.name,
                "version": reference.version,
                "fingerprint": reference.fingerprint,
            }
            for reference in bank.prompt_references
        ]

        return (
            json.dumps(questions, ensure_ascii=False),
            json.dumps(prompts, ensure_ascii=False),
        )

    @staticmethod
    def _deserialize(
        *,
        document_id: UUID,
        identity_fingerprint: str,
        row: sqlite3.Row,
    ) -> QuestionBank:
        questions = tuple(
            StudyQuestion(
                number=item["number"],
                text=item["text"],
                expected_answer=item["expected_answer"],
                citations=tuple(
                    Citation(
                        number=citation["number"],
                        source=citation["source"],
                        page_number=citation["page_number"],
                        chunk_index=citation["chunk_index"],
                        excerpt=citation["excerpt"],
                    )
                    for citation in item["citations"]
                ),
            )
            for item in json.loads(row["questions_json"])
        )
        prompts = tuple(
            PromptReference(
                name=item["name"],
                version=item["version"],
                fingerprint=item["fingerprint"],
            )
            for item in json.loads(row["prompt_references_json"])
        )

        return QuestionBank(
            document_id=document_id,
            identity_fingerprint=identity_fingerprint,
            source=row["source"],
            questions=questions,
            prompt_references=prompts,
        )
