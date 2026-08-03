"""Application service for indexing and searching documents."""

from collections.abc import Sequence
from typing import Protocol

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


class DocumentSearchService:
    """Coordinate document chunking and retrieval indexing."""

    def __init__(
        self,
        chunker: TextChunker,
        retrieval: RetrievalGateway,
    ) -> None:
        self.chunker = chunker
        self.retrieval = retrieval

    def index_document(self, document: Document) -> list[Chunk]:
        """Chunk a document, index all chunks, and return them."""

        chunks = self.chunker.chunk_pages(document.pages)
        self.retrieval.index_chunks(chunks)
        return chunks

    def search(self, query: str, limit: int) -> list[SearchResult]:
        """Return chunks relevant to a search query."""

        return self.retrieval.search(query, limit=limit)
