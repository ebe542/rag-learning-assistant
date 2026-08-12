"""Command execution and dependency wiring for the CLI."""

import json
import sys
from collections.abc import Sequence
from pathlib import Path
from uuid import UUID

from rag_learning_assistant.application import (
    BatchImportService,
    DocumentSearchService,
    DocumentSummarizationService,
    ImportOutcome,
    ImportStatus,
    LibraryCatalog,
    LibraryService,
    QuestionAnsweringService,
)
from rag_learning_assistant.application.summarization import (
    SUMMARY_MAP_PROMPT,
    SUMMARY_REDUCE_PROMPT,
)
from rag_learning_assistant.chunking import TextChunker
from rag_learning_assistant.generation import (
    GenerationIdentity,
    HuggingFaceTextGenerator,
    PromptReference,
)
from rag_learning_assistant.generation.huggingface import (
    JSON_REPAIR_PROMPT,
    SYSTEM_PROMPT,
)
from rag_learning_assistant.generation.sqlite_cache import SqliteSummaryCache
from rag_learning_assistant.ingestion import Document, PdfExtractor
from rag_learning_assistant.library import IndexedDocument, SqliteDocumentRepository
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


def build_question_answering_service(
    index_directory: Path,
) -> QuestionAnsweringService:
    """Build source-grounded question answering for a persistent index."""

    return QuestionAnsweringService(
        search=build_persistent_retrieval(index_directory),
        generator=HuggingFaceTextGenerator(),
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


def write_summarization_progress(
    phase: str,
    current: int,
    total: int,
) -> None:
    """Write human-readable summarization progress without corrupting JSON."""

    if phase == "map":
        message = f"Summarizing batch {current}/{total}..."
    else:
        message = "Combining partial summaries..."

    # Flush immediately because each following model call may take minutes.
    print(message, file=sys.stderr, flush=True)


def build_document_summarization_service(
    index_directory: Path,
    max_map_new_tokens: int,
    max_reduce_new_tokens: int,
    max_batch_chars: int,
) -> DocumentSummarizationService:
    """Build document-wide summarization for a persistent library."""

    # The vector store validates that its embedding metadata matches the
    # existing index, even though summarization reads only stored chunks.
    embedder = SentenceTransformerEmbedder()
    store = FaissVectorStore(
        index_directory,
        model_name=embedder.model_name,
        model_revision=embedder.model_revision,
    )
    database_path = index_directory / "metadata.sqlite3"
    repository = SqliteDocumentRepository(database_path)
    generator = HuggingFaceTextGenerator(
        max_new_tokens=max_reduce_new_tokens,
    )

    def build_identity(document: IndexedDocument) -> GenerationIdentity:
        # Every input that can change a partial summary belongs in the cache key.
        # Including the repair prompt also makes repaired responses reproducible.
        return GenerationIdentity(
            model_name=generator.model_name,
            model_revision=generator.model_revision,
            prompt_references=(
                SUMMARY_MAP_PROMPT.reference,
                SUMMARY_REDUCE_PROMPT.reference,
                SYSTEM_PROMPT.reference,
                JSON_REPAIR_PROMPT.reference,
            ),
            max_map_new_tokens=max_map_new_tokens,
            max_reduce_new_tokens=max_reduce_new_tokens,
            max_batch_chars=max_batch_chars,
            document_content_sha256=document.content_sha256,
        )

    return DocumentSummarizationService(
        documents=repository,
        chunks=store,
        generator=generator,
        max_batch_chars=max_batch_chars,
        max_map_new_tokens=max_map_new_tokens,
        max_reduce_new_tokens=max_reduce_new_tokens,
        progress=write_summarization_progress,
        cache=SqliteSummaryCache(database_path),
        identity_factory=build_identity,
    )


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


def run_summarize(
    index_directory: Path,
    document_id: UUID,
    max_map_new_tokens: int,
    max_reduce_new_tokens: int,
    max_batch_chars: int,
) -> int:
    """Summarize one indexed document and write the result as JSON."""

    service = build_document_summarization_service(
        index_directory,
        max_map_new_tokens,
        max_reduce_new_tokens,
        max_batch_chars,
    )
    summary = service.summarize(document_id)

    payload = {
        "document_id": str(summary.document_id),
        "source": summary.source,
        "summary": summary.text,
        # Citation metadata is reconstructed from persistent chunks rather
        # than accepted from model-generated text.
        "citations": [
            {
                "number": citation.number,
                "source": citation.source,
                "page_number": citation.page_number,
                "chunk_index": citation.chunk_index,
                "excerpt": citation.excerpt,
            }
            for citation in summary.citations
        ],
        "prompts": _serialize_prompt_references(summary.prompt_references),
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


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


def run_ask(
    index_directory: Path,
    question: str,
    limit: int,
) -> int:
    """Answer a question and write the grounded result as JSON."""

    answerer = build_question_answering_service(index_directory)
    answer = answerer.answer(question, limit=limit)

    payload = {
        "question": answer.question,
        "answer": answer.text,
        # Citation metadata comes from retrieval, not from model-generated text.
        # This keeps source references trustworthy even if the model misbehaves.
        "citations": [
            {
                "number": citation.number,
                "source": citation.source,
                "page_number": citation.page_number,
                "chunk_index": citation.chunk_index,
                "excerpt": citation.excerpt,
            }
            for citation in answer.citations
        ],
        "prompts": _serialize_prompt_references(answer.prompt_references),
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


def _serialize_prompt_references(
    references: Sequence[PromptReference],
) -> list[dict[str, object]]:
    """Convert prompt identities without exposing their full text."""

    return [
        {
            "name": reference.name,
            "version": reference.version,
            "fingerprint": reference.fingerprint,
        }
        for reference in references
    ]


def run_replace(
    document_id: UUID,
    pdf_path: Path,
    chunker: TextChunker,
    index_directory: Path,
) -> int:
    """Replace one document and write its updated metadata as JSON."""

    library = build_library_service(
        chunker,
        index_directory,
    )
    document = library.replace_document(
        document_id,
        pdf_path,
    )

    payload = {
        "index_directory": str(index_directory),
        "replaced_document": {
            "id": str(document.id),
            "source": document.source,
            "content_sha256": document.content_sha256,
            "page_count": document.page_count,
            "chunk_count": document.chunk_count,
        },
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
