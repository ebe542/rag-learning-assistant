"""Application service for indexing and retrieving chunks."""

from collections.abc import Sequence
from uuid import UUID

from rag_learning_assistant.chunking import Chunk
from rag_learning_assistant.retrieval.embeddings import Embedder
from rag_learning_assistant.retrieval.models import SearchResult
from rag_learning_assistant.retrieval.store import VectorStore


class RetrievalService:
    """Coordinate text embedding, storage, and similarity search."""

    def __init__(
        self,
        embedder: Embedder,
        store: VectorStore,
    ) -> None:
        self.embedder = embedder
        self.store = store

    def index_chunks(self, chunks: Sequence[Chunk]) -> None:
        """Embed and store a batch of chunks."""

        embeddings = self.embedder.embed_documents([chunk.text for chunk in chunks])

        if len(embeddings) != len(chunks):
            raise ValueError("Embedder must return one embedding per chunk")

        entries = list(zip(chunks, embeddings, strict=True))
        self.store.add_many(entries)

    def remove_document(self, document_id: UUID) -> int:
        """Remove all stored chunks belonging to a document."""

        return self.store.remove_document(document_id)

    def search(self, query: str, limit: int) -> list[SearchResult]:
        """Embed a query and return the most similar chunks."""

        query_embedding = self.embedder.embed_query(query)

        return self.store.search(query_embedding, limit=limit)
