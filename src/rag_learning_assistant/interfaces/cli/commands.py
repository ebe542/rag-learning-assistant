"""Command execution and dependency wiring for the CLI."""

import json
from pathlib import Path

from rag_learning_assistant.application import DocumentSearchService
from rag_learning_assistant.chunking import TextChunker
from rag_learning_assistant.ingestion import Document
from rag_learning_assistant.retrieval import (
    FaissVectorStore,
    RetrievalService,
    SentenceTransformerEmbedder,
)


def build_persistent_retrieval(
    index_directory: Path,
) -> RetrievalService:
    """Build retrieval backed by an existing persistent index."""

    embedder = SentenceTransformerEmbedder()
    store = FaissVectorStore(
        index_directory,
        model_name=embedder.model_name,
        model_revision=embedder.model_revision,
    )
    return RetrievalService(
        embedder=embedder,
        store=store,
    )


def build_persistent_document_search(
    chunker: TextChunker,
    index_directory: Path,
) -> DocumentSearchService:
    """Build document indexing backed by a persistent FAISS index."""

    return DocumentSearchService(
        chunker=chunker,
        retrieval=build_persistent_retrieval(index_directory),
    )


def run_index(
    document: Document,
    chunker: TextChunker,
    index_directory: Path,
) -> int:
    """Persist the searchable chunks of a document."""

    search = build_persistent_document_search(
        chunker,
        index_directory,
    )
    chunks = search.index_document(document)

    payload = {
        "source": document.source,
        "index_directory": str(index_directory),
        "chunks_indexed": len(chunks),
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


def run_extract(
    document: Document,
    chunker: TextChunker,
) -> int:
    """Write extracted pages and chunks as JSON."""

    chunks = chunker.chunk_pages(document.pages)
    payload = {
        "source": document.source,
        "pages": [
            {
                "number": page.number,
                "source": page.source,
                "text": page.text,
            }
            for page in document.pages
        ],
        "chunks": [
            {
                "index": chunk.index,
                "text": chunk.text,
                "source": chunk.source,
                "page_number": chunk.page_number,
            }
            for chunk in chunks
        ],
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


def run_search(
    index_directory: Path,
    query: str,
    limit: int,
) -> int:
    """Search an existing index and write ranked results as JSON."""

    retrieval = build_persistent_retrieval(index_directory)
    results = retrieval.search(query, limit=limit)

    payload = {
        "query": query,
        "results": [
            {
                "score": result.score,
                "text": result.chunk.text,
                "source": result.chunk.source,
                "page_number": result.chunk.page_number,
                "index": result.chunk.index,
            }
            for result in results
        ],
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0
