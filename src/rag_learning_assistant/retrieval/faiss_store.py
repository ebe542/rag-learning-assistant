"""Persistent vector storage using FAISS and SQLite."""

import sqlite3
from collections.abc import Sequence
from pathlib import Path
from typing import Any
from uuid import UUID

from rag_learning_assistant.chunking import Chunk
from rag_learning_assistant.retrieval.embeddings import Embedding
from rag_learning_assistant.retrieval.models import SearchResult


class FaissVectorStore:
    """Persist embeddings in FAISS and chunk metadata in SQLite."""

    def __init__(
        self,
        index_directory: str | Path,
        *,
        model_name: str,
        model_revision: str,
    ) -> None:
        if not model_name.strip():
            raise ValueError("Model name must not be blank")

        if not model_revision.strip():
            raise ValueError("Model revision must not be blank")

        self.index_directory = Path(index_directory)
        self.index_path = self.index_directory / "vectors.faiss"
        self.metadata_path = self.index_directory / "metadata.sqlite3"

        self.index_directory.mkdir(parents=True, exist_ok=True)
        self._initialize_database()
        self._initialize_model_metadata(model_name, model_revision)

    def add(self, chunk: Chunk, embedding: Embedding) -> None:
        """Persist one embedding and its associated chunk."""

        self.add_many([(chunk, embedding)])

    def add_many(
        self,
        entries: Sequence[tuple[Chunk, Embedding]],
    ) -> None:
        """Persist multiple embeddings and their chunks in one batch."""

        if not entries:
            return

        dimension = len(entries[0][1])

        # Validate the complete batch before changing SQLite or FAISS.
        for _, embedding in entries:
            if not embedding or not any(value != 0.0 for value in embedding):
                raise ValueError("Embedding must not be a zero vector")

            if len(embedding) != dimension:
                raise ValueError(f"Embedding dimension must be {dimension}")

        self._validate_or_store_dimension(dimension)

        faiss, numpy = self._load_vector_libraries()
        index = self._load_or_create_index(dimension, faiss)

        if index.d != dimension:
            raise ValueError(f"Embedding dimension must be {index.d}")

        entry_ids: list[int] = []

        with sqlite3.connect(self.metadata_path) as connection:
            for chunk, _ in entries:
                cursor = connection.execute(
                    """
                    INSERT INTO chunks (
                        text,
                        source,
                        page_number,
                        chunk_index,
                        document_id
                    )
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (
                        chunk.text,
                        chunk.source,
                        chunk.page_number,
                        chunk.index,
                        (str(chunk.document_id) if chunk.document_id is not None else None),
                    ),
                )
                entry_id = cursor.lastrowid

                if entry_id is None:
                    raise RuntimeError("SQLite did not create a chunk ID")

                entry_ids.append(entry_id)

            vectors = numpy.asarray(
                [embedding for _, embedding in entries],
                dtype="float32",
            )

            # Inner product equals cosine similarity for normalized vectors.
            faiss.normalize_L2(vectors)
            index.add_with_ids(
                vectors,
                numpy.asarray(entry_ids, dtype="int64"),
            )

            # Write once per batch instead of once per chunk.
            faiss.write_index(index, str(self.index_path))

    def remove_document(self, document_id: UUID) -> int:
        """Remove all vectors and chunk metadata belonging to a document."""

        with sqlite3.connect(self.metadata_path) as connection:
            rows = connection.execute(
                """
                SELECT id
                FROM chunks
                WHERE document_id = ?
                """,
                (str(document_id),),
            ).fetchall()

        entry_ids = [row[0] for row in rows]

        if not entry_ids:
            return 0

        if not self.index_path.is_file():
            raise RuntimeError("FAISS index is missing for stored chunks")

        faiss, numpy = self._load_vector_libraries()
        index = faiss.read_index(str(self.index_path))
        id_array = numpy.asarray(entry_ids, dtype="int64")
        removed_count = int(index.remove_ids(id_array))

        if removed_count != len(entry_ids):
            raise RuntimeError("FAISS and SQLite contain different document chunks")

        temporary_index_path = self.index_path.with_suffix(".faiss.tmp")

        try:
            faiss.write_index(index, str(temporary_index_path))

            with sqlite3.connect(self.metadata_path) as connection:
                connection.execute(
                    """
                    DELETE FROM chunks
                    WHERE document_id = ?
                    """,
                    (str(document_id),),
                )

                # Replace the persisted index only after both storage operations
                # have succeeded up to this point.
                temporary_index_path.replace(self.index_path)
        finally:
            temporary_index_path.unlink(missing_ok=True)

        return removed_count

    def search(self, query: Embedding, limit: int) -> list[SearchResult]:
        """Return the most similar persisted chunks."""

        if limit < 1:
            raise ValueError("limit must be positive")

        if not query or not any(value != 0.0 for value in query):
            raise ValueError("Embedding must not be a zero vector")

        if not self.index_path.exists():
            return []

        faiss, numpy = self._load_vector_libraries()
        index = faiss.read_index(str(self.index_path))

        if index.d != len(query):
            raise ValueError(f"Embedding dimension must be {index.d}")

        query_vector = numpy.asarray([query], dtype="float32")

        # Stored vectors and queries must use the same normalization.
        faiss.normalize_L2(query_vector)
        result_count = min(limit, index.ntotal)

        if result_count == 0:
            return []

        scores, entry_ids = index.search(query_vector, result_count)

        ordered_ids = [int(entry_id) for entry_id in entry_ids[0]]

        with sqlite3.connect(self.metadata_path) as connection:
            placeholders = ", ".join("?" for _ in ordered_ids)
            rows = connection.execute(
                f"""
                SELECT
                    id,
                    text,
                    source,
                    page_number,
                    chunk_index,
                    document_id
                FROM chunks
                WHERE id IN ({placeholders})
                """,
                ordered_ids,
            ).fetchall()

        chunks_by_id = {
            row[0]: Chunk(
                text=row[1],
                source=row[2],
                page_number=row[3],
                index=row[4],
                document_id=(UUID(row[5]) if row[5] is not None else None),
            )
            for row in rows
        }

        results: list[SearchResult] = []

        # SQL does not preserve the order of an IN query, so rebuild FAISS order.
        for entry_id, score in zip(ordered_ids, scores[0], strict=True):
            chunk = chunks_by_id.get(entry_id)

            if chunk is None:
                raise RuntimeError(f"Missing metadata for FAISS ID {entry_id}")

            results.append(
                SearchResult(
                    chunk=chunk,
                    score=float(score),
                )
            )

        return results

    def _initialize_database(self) -> None:
        """Create the metadata schema when opening a new index directory."""

        with sqlite3.connect(self.metadata_path) as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS chunks (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    text TEXT NOT NULL,
                    source TEXT NOT NULL,
                    page_number INTEGER NOT NULL,
                    chunk_index INTEGER NOT NULL,
                    document_id TEXT
                )
                """
            )
            self._ensure_document_id_column(connection)
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS index_metadata (
                    id INTEGER PRIMARY KEY CHECK (id = 1),
                    model_name TEXT NOT NULL,
                    model_revision TEXT NOT NULL,
                    embedding_dimension INTEGER
                )
                """
            )

    @staticmethod
    def _ensure_document_id_column(
        connection: sqlite3.Connection,
    ) -> None:
        """Add document IDs to indices created before library support."""

        column_names = {row[1] for row in connection.execute("PRAGMA table_info(chunks)")}

        if "document_id" not in column_names:
            connection.execute("ALTER TABLE chunks ADD COLUMN document_id TEXT")

    def _initialize_model_metadata(
        self,
        model_name: str,
        model_revision: str,
    ) -> None:
        """Create model metadata or verify an existing index."""

        with sqlite3.connect(self.metadata_path) as connection:
            stored_metadata = connection.execute(
                """
                SELECT model_name, model_revision
                FROM index_metadata
                WHERE id = 1
                """
            ).fetchone()

            if stored_metadata is None:
                connection.execute(
                    """
                    INSERT INTO index_metadata (
                        id,
                        model_name,
                        model_revision,
                        embedding_dimension
                    )
                    VALUES (1, ?, ?, NULL)
                    """,
                    (model_name, model_revision),
                )
                return

            if stored_metadata != (model_name, model_revision):
                raise ValueError("Index was created with a different embedding model")

    def _validate_or_store_dimension(self, dimension: int) -> None:
        """Store the first vector dimension and validate later vectors."""

        with sqlite3.connect(self.metadata_path) as connection:
            stored_row = connection.execute(
                """
                SELECT embedding_dimension
                FROM index_metadata
                WHERE id = 1
                """
            ).fetchone()

            if stored_row is None:
                raise RuntimeError("Index model metadata is missing")

            stored_dimension = stored_row[0]

            if stored_dimension is None:
                connection.execute(
                    """
                    UPDATE index_metadata
                    SET embedding_dimension = ?
                    WHERE id = 1
                    """,
                    (dimension,),
                )
                return

            if stored_dimension != dimension:
                raise ValueError(f"Embedding dimension must be {stored_dimension}")

    def _load_or_create_index(self, dimension: int, faiss: Any) -> Any:
        """Load an existing FAISS index or create an exact inner-product index."""

        if self.index_path.exists():
            return faiss.read_index(str(self.index_path))

        return faiss.IndexIDMap2(faiss.IndexFlatIP(dimension))

    @staticmethod
    def _load_vector_libraries() -> tuple[Any, Any]:
        """Load optional storage dependencies only when the store is used."""

        import faiss
        import numpy

        return faiss, numpy
