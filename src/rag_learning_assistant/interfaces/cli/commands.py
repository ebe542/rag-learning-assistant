"""Command execution and dependency wiring for the CLI."""

import json
from collections.abc import Sequence
from pathlib import Path
from uuid import UUID

from rag_learning_assistant.application import (
    BatchImportService,
    DocumentSearchService,
    ImportOutcome,
    ImportStatus,
    LibraryCatalog,
    LibraryService,
)
from rag_learning_assistant.chunking import TextChunker
from rag_learning_assistant.ingestion import Document, PdfExtractor
from rag_learning_assistant.library import SqliteDocumentRepository
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


def build_library_service(
    chunker: TextChunker,
    index_directory: Path,
) -> LibraryService:
    """Build document-library management for one persistent index."""

    repository = SqliteDocumentRepository(index_directory / "metadata.sqlite3")
    return LibraryService(
        repository=repository,
        extractor=PdfExtractor(),
        indexer=build_persistent_document_search(
            chunker,
            index_directory,
        ),
    )


def build_library_catalog(
    index_directory: Path,
) -> LibraryCatalog:
    """Build read-only access to persistent library metadata."""

    repository = SqliteDocumentRepository(index_directory / "metadata.sqlite3")
    return LibraryCatalog(repository)


def run_index(
    pdf_paths: Sequence[Path],
    chunker: TextChunker,
    index_directory: Path,
) -> int:
    """Index and register documents in a persistent library."""

    library = build_library_service(
        chunker,
        index_directory,
    )
    outcomes = BatchImportService(library).add_documents(pdf_paths)

    payload = {
        "index_directory": str(index_directory),
        "results": [_serialize_import_outcome(outcome) for outcome in outcomes],
        "summary": {
            "added": sum(outcome.status is ImportStatus.ADDED for outcome in outcomes),
            "skipped": sum(outcome.status is ImportStatus.SKIPPED for outcome in outcomes),
            "failed": sum(outcome.status is ImportStatus.FAILED for outcome in outcomes),
        },
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 1 if any(outcome.status is ImportStatus.FAILED for outcome in outcomes) else 0


def _serialize_import_outcome(outcome: ImportOutcome) -> dict[str, object]:
    """Convert one batch result into a JSON-compatible mapping."""

    document = outcome.document
    document_payload = (
        {
            "id": str(document.id),
            "source": document.source,
            "content_sha256": document.content_sha256,
            "page_count": document.page_count,
            "chunk_count": document.chunk_count,
        }
        if document is not None
        else None
    )

    return {
        "path": str(outcome.path),
        "status": outcome.status.value,
        "document": document_payload,
        "message": outcome.message,
    }


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


def run_remove(
    document_id: UUID,
    chunker: TextChunker,
    index_directory: Path,
) -> int:
    """Remove one document and write its former metadata as JSON."""

    library = build_library_service(
        chunker,
        index_directory,
    )
    document = library.remove_document(document_id)

    payload = {
        "index_directory": str(index_directory),
        "removed_document": {
            "id": str(document.id),
            "source": document.source,
            "content_sha256": document.content_sha256,
            "page_count": document.page_count,
            "chunk_count": document.chunk_count,
        },
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


def run_list(index_directory: Path) -> int:
    """Write all registered library documents as JSON."""

    catalog = build_library_catalog(index_directory)
    documents = catalog.list_documents()

    payload = {
        "index_directory": str(index_directory),
        "documents": [
            {
                "id": str(document.id),
                "source": document.source,
                "content_sha256": document.content_sha256,
                "page_count": document.page_count,
                "chunk_count": document.chunk_count,
            }
            for document in documents
        ],
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0
