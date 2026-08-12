"""SQLite persistence for resumable summary generation."""

import json
import sqlite3
from contextlib import closing
from pathlib import Path

from rag_learning_assistant.generation.cache import (
    CachedSummaryBatch,
)
from rag_learning_assistant.generation.models import (
    GenerationResult,
)
from rag_learning_assistant.generation.prompts import (
    PromptReference,
)


class SqliteSummaryCache:
    """Persist successful map results in the library database."""

    def __init__(self, database_path: str | Path) -> None:
        self.database_path = Path(database_path)
        self.database_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )
        self._initialize_database()

    def save_batch(
        self,
        batch: CachedSummaryBatch,
    ) -> None:
        """Persist one successfully generated summary batch."""

        citation_numbers = json.dumps(
            batch.result.citation_numbers,
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

        with closing(sqlite3.connect(self.database_path)) as connection:
            connection.execute(
                """
                INSERT INTO summary_batches (
                    identity_fingerprint,
                    batch_number,
                    first_context_number,
                    last_context_number,
                    summary_text,
                    citation_numbers,
                    prompt_references
                )
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT (
                    identity_fingerprint,
                    batch_number
                )
                DO NOTHING
                """,
                (
                    batch.identity_fingerprint,
                    batch.batch_number,
                    batch.first_context_number,
                    batch.last_context_number,
                    batch.result.text,
                    citation_numbers,
                    prompt_references,
                ),
            )
            connection.commit()

        stored = self.find_batch(
            identity_fingerprint=batch.identity_fingerprint,
            batch_number=batch.batch_number,
        )

        if stored is None:
            raise RuntimeError("Cached summary batch was not persisted")

        # Retrying after an interruption may write the same completed batch again.
        # Identical data is safe; different data for the same cache key indicates
        # nondeterministic or corrupted state and must never be overwritten.
        if stored != batch:
            raise ValueError("Cached summary batch conflicts with existing data")

    def find_batch(
        self,
        identity_fingerprint: str,
        batch_number: int,
    ) -> CachedSummaryBatch | None:
        """Return one cached batch for a generation identity."""

        with closing(sqlite3.connect(self.database_path)) as connection:
            row = connection.execute(
                """
                SELECT
                    first_context_number,
                    last_context_number,
                    summary_text,
                    citation_numbers,
                    prompt_references
                FROM summary_batches
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
            first_context_number,
            last_context_number,
            summary_text,
            citation_numbers_json,
            prompt_references_json,
        ) = row

        prompt_references = tuple(
            PromptReference(
                name=value["name"],
                version=value["version"],
                fingerprint=value["fingerprint"],
            )
            for value in json.loads(prompt_references_json)
        )

        return CachedSummaryBatch(
            identity_fingerprint=identity_fingerprint,
            batch_number=batch_number,
            first_context_number=first_context_number,
            last_context_number=last_context_number,
            result=GenerationResult(
                text=summary_text,
                citation_numbers=tuple(json.loads(citation_numbers_json)),
                prompt_references=prompt_references,
            ),
        )

    def _initialize_database(self) -> None:
        """Create the cache table without changing existing library data."""

        with closing(sqlite3.connect(self.database_path)) as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS summary_batches (
                    identity_fingerprint TEXT NOT NULL,
                    batch_number INTEGER NOT NULL,
                    first_context_number INTEGER NOT NULL,
                    last_context_number INTEGER NOT NULL,
                    summary_text TEXT NOT NULL,
                    citation_numbers TEXT NOT NULL,
                    prompt_references TEXT NOT NULL,
                    PRIMARY KEY (
                        identity_fingerprint,
                        batch_number
                    )
                )
                """
            )
            connection.commit()
