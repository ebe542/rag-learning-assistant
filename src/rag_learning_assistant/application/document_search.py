"""Application service for indexing and searching documents."""

from collections.abc import Sequence
from dataclasses import replace
from typing import Protocol
from uuid import UUID

from rag_learning_assistant.chunking import Chunk, TextChunker
from rag_learning_assistant.ingestion import Document
from rag_learning_assistant.retrieval import SearchResult


class RetrievalGateway(Protocol):
    """Index chunks and retrieve relevant results."""

    def index_chunks(self, chunks: Sequence[Chunk]) -> None:
        """Embed and store a batch of chunks."""
        ...

    def search(self, query: str, limit: int) -> list[SearchResult]:
        """Return chunks relevant to a query."""
        ...

    def remove_document(self, document_id: UUID) -> int:
        """Remove all indexed chunks belonging to a document."""

        ...

    def replace_document(
        self,
        document_id: UUID,
        chunks: Sequence[Chunk],
    ) -> None:
        """Replace all indexed chunks belonging to a document."""

        ...


class DocumentSearchService:
    """Coordinate document chunking and retrieval indexing."""

    def __init__(
        self,
        chunker: TextChunker,
        retrieval: RetrievalGateway,
    ) -> None:
        self.chunker = chunker
        self.retrieval = retrieval

    def index_document(
        self,
        document: Document,
        *,
        document_id: UUID | None = None,
    ) -> list[Chunk]:
        """Chunk a document, index all chunks, and return them."""

        chunks = self.chunker.chunk_pages(document.pages)

        if document_id is not None:
            # Chunks are immutable, so library metadata is added by copying.
            chunks = [replace(chunk, document_id=document_id) for chunk in chunks]

        self.retrieval.index_chunks(chunks)
        return chunks

    def replace_document(
        self,
        document: Document,
        document_id: UUID,
    ) -> list[Chunk]:
        """Chunk a replacement document while preserving its ID."""

        chunks = [
            replace(chunk, document_id=document_id)
            for chunk in self.chunker.chunk_pages(document.pages)
        ]
        self.retrieval.replace_document(
            document_id,
            chunks,
        )
        return chunks

    def remove_document(self, document_id: UUID) -> int:
        """Remove all searchable chunks belonging to a document."""

        return self.retrieval.remove_document(document_id)

    def search(self, query: str, limit: int) -> list[SearchResult]:
        """Return chunks relevant to a search query."""

        return self.retrieval.search(query, limit=limit)
