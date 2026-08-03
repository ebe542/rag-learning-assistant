"""Semantic retrieval of relevant document chunks."""

from rag_learning_assistant.retrieval.embeddings import Embedder, Embedding
from rag_learning_assistant.retrieval.faiss_store import FaissVectorStore
from rag_learning_assistant.retrieval.models import SearchResult
from rag_learning_assistant.retrieval.sentence_transformer import (
    SentenceTransformerEmbedder,
)
from rag_learning_assistant.retrieval.service import RetrievalService
from rag_learning_assistant.retrieval.store import InMemoryVectorStore, VectorStore

__all__ = [
    "Embedder",
    "Embedding",
    "FaissVectorStore",
    "InMemoryVectorStore",
    "RetrievalService",
    "SearchResult",
    "SentenceTransformerEmbedder",
    "VectorStore",
]
