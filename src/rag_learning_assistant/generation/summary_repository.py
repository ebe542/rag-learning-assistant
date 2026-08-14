import json
import sqlite3
from contextlib import closing
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol
from uuid import UUID

from rag_learning_assistant.generation.models import Citation
from rag_learning_assistant.generation.prompts import PromptReference


@dataclass(frozen=True, slots=True)
class PersistedDocumentSummary:
    """Store one final summary together with its generation identity."""

    document_id: UUID
    identity_fingerprint: str
    source: str
    text: str
    citations: tuple[Citation, ...]
    prompt_references: tuple[PromptReference, ...]

    def __post_init__(self) -> None:
        if not self.identity_fingerprint.strip():
            raise ValueError("Persisted summary identity_fingerprint must not be blank")

        if not self.source.strip():
            raise ValueError("Persisted summary source must not be blank")

        if not self.text.strip():
            raise ValueError("Persisted summary text must not be blank")

        if not self.citations:
            raise ValueError("Persisted summary requires at least one citation")

        citation_numbers = [citation.number for citation in self.citations]
        if len(set(citation_numbers)) != len(citation_numbers):
            raise ValueError("Persisted summary citation numbers must be unique")

        if not self.prompt_references:
            raise ValueError("Persisted summary requires at least one prompt reference")


class DocumentSummaryRepository(Protocol):
    """Store and retrieve final summaries by their complete generation identity."""

    def find(
        self,
        document_id: UUID,
        identity_fingerprint: str,
    ) -> PersistedDocumentSummary | None: ...

    def save(self, summary: PersistedDocumentSummary) -> None: ...

    def replace(self, summary: PersistedDocumentSummary) -> None: ...


class SqliteDocumentSummaryRepository:
    """Persist final document summaries by document and generation identity."""

    def __init__(self, database_path: Path) -> None:
        self.database_path = database_path
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize_schema()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.database_path)
        connection.row_factory = sqlite3.Row
        return connection

    def _initialize_schema(self) -> None:
        # A sqlite connection context manages the transaction, but it does not
        # close the connection. ``closing`` makes both responsibilities explicit.
        with closing(self._connect()) as connection, connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS document_summaries (
                    document_id TEXT NOT NULL,
                    identity_fingerprint TEXT NOT NULL,
                    source TEXT NOT NULL,
                    text TEXT NOT NULL,
                    citations_json TEXT NOT NULL,
                    prompt_references_json TEXT NOT NULL,
                    PRIMARY KEY (document_id, identity_fingerprint)
                )
                """
            )

    def save(self, summary: PersistedDocumentSummary) -> None:
        existing = self.find(
            summary.document_id,
            summary.identity_fingerprint,
        )

        if existing == summary:
            return

        if existing is not None:
            raise ValueError("Conflicting final summary already exists")

        citations_json, prompt_references_json = self._serialize(summary)

        with closing(self._connect()) as connection, connection:
            connection.execute(
                """
                INSERT INTO document_summaries (
                    document_id,
                    identity_fingerprint,
                    source,
                    text,
                    citations_json,
                    prompt_references_json
                )
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    str(summary.document_id),
                    summary.identity_fingerprint,
                    summary.source,
                    summary.text,
                    citations_json,
                    prompt_references_json,
                ),
            )

    def find(
        self,
        document_id: UUID,
        identity_fingerprint: str,
    ) -> PersistedDocumentSummary | None:
        with closing(self._connect()) as connection:
            row = connection.execute(
                """
                SELECT
                    source,
                    text,
                    citations_json,
                    prompt_references_json
                FROM document_summaries
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

    def replace(self, summary: PersistedDocumentSummary) -> None:
        """Explicitly replace a final result for the same generation identity."""

        citations_json, prompt_references_json = self._serialize(summary)

        with closing(self._connect()) as connection, connection:
            cursor = connection.execute(
                """
                UPDATE document_summaries
                SET source = ?,
                    text = ?,
                    citations_json = ?,
                    prompt_references_json = ?
                WHERE document_id = ?
                  AND identity_fingerprint = ?
                """,
                (
                    summary.source,
                    summary.text,
                    citations_json,
                    prompt_references_json,
                    str(summary.document_id),
                    summary.identity_fingerprint,
                ),
            )

        if cursor.rowcount == 0:
            # Force regeneration can also be the first completed run when no
            # final result exists yet.
            self.save(summary)

    def delete_document(self, document_id: UUID) -> int:
        """Delete every persisted summary version belonging to one document."""

        with closing(self._connect()) as connection, connection:
            cursor = connection.execute(
                """
                DELETE FROM document_summaries
                WHERE document_id = ?
                """,
                (str(document_id),),
            )

            return cursor.rowcount

    def list_document(
        self,
        document_id: UUID,
    ) -> list[PersistedDocumentSummary]:
        """Return every persisted summary version for one document."""

        with closing(self._connect()) as connection:
            rows = connection.execute(
                """
                SELECT
                    identity_fingerprint,
                    source,
                    text,
                    citations_json,
                    prompt_references_json
                FROM document_summaries
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

    @staticmethod
    def _serialize(
        summary: PersistedDocumentSummary,
    ) -> tuple[str, str]:
        citations = [
            {
                "number": citation.number,
                "source": citation.source,
                "page_number": citation.page_number,
                "chunk_index": citation.chunk_index,
                "excerpt": citation.excerpt,
            }
            for citation in summary.citations
        ]
        prompt_references = [
            {
                "name": reference.name,
                "version": reference.version,
                "fingerprint": reference.fingerprint,
            }
            for reference in summary.prompt_references
        ]
        return (
            json.dumps(citations, ensure_ascii=False),
            json.dumps(prompt_references, ensure_ascii=False),
        )

    @staticmethod
    def _deserialize(
        *,
        document_id: UUID,
        identity_fingerprint: str,
        row: sqlite3.Row,
    ) -> PersistedDocumentSummary:
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
        prompt_references = tuple(
            PromptReference(
                name=item["name"],
                version=item["version"],
                fingerprint=item["fingerprint"],
            )
            for item in json.loads(row["prompt_references_json"])
        )

        return PersistedDocumentSummary(
            document_id=document_id,
            identity_fingerprint=identity_fingerprint,
            source=row["source"],
            text=row["text"],
            citations=citations,
            prompt_references=prompt_references,
        )
