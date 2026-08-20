"""Command execution and dependency wiring for the CLI."""

import json
import sys
from collections.abc import Callable, Sequence
from datetime import UTC, datetime
from pathlib import Path
from uuid import UUID, uuid4

from rag_learning_assistant.application import (
    AnswerEvaluationService,
    BatchImportService,
    DocumentSearchService,
    DocumentSummarizationService,
    DocumentSummaryCatalog,
    ImportOutcome,
    ImportStatus,
    LearningPackageCatalog,
    LearningPackageService,
    LearningPackageStudyService,
    LibraryCatalog,
    LibraryService,
    QuestionAnsweringService,
    QuestionBankCatalog,
    QuestionBankService,
    ReviewScheduler,
    ReviewService,
    StudySessionService,
)
from rag_learning_assistant.application.question_bank import (
    QUESTION_BANK_PROMPT,
)
from rag_learning_assistant.application.summarization import (
    SUMMARY_MAP_PROMPT,
    SUMMARY_REDUCE_PROMPT,
)
from rag_learning_assistant.chunking import TextChunker
from rag_learning_assistant.generation import (
    Citation,
    GenerationIdentity,
    HuggingFaceTextGenerator,
    PersistedDocumentSummary,
    PromptReference,
    SqliteDocumentSummaryRepository,
)
from rag_learning_assistant.generation.huggingface import (
    JSON_REPAIR_PROMPT,
    QUESTION_JSON_REPAIR_PROMPT,
    QUESTION_SYSTEM_PROMPT,
    SYSTEM_PROMPT,
)
from rag_learning_assistant.generation.sqlite_cache import SqliteSummaryCache
from rag_learning_assistant.ingestion import Document, PdfExtractor
from rag_learning_assistant.interfaces.cli.parsing import (
    DEFAULT_ANSWER_EVALUATION_MAX_NEW_TOKENS,
    DEFAULT_MAX_CHARS,
    DEFAULT_OVERLAP_CHARS,
    DEFAULT_QUESTION_MAX_NEW_TOKENS,
    DEFAULT_SUMMARY_MAX_BATCH_CHARS,
    DEFAULT_SUMMARY_MAX_MAP_NEW_TOKENS,
    DEFAULT_SUMMARY_MAX_REDUCE_NEW_TOKENS,
)
from rag_learning_assistant.interfaces.cli.study import (
    capture_study_answer,
)
from rag_learning_assistant.learning import (
    LearningPackage,
    QuestionBankIdentity,
    QuestionProgress,
    ReviewRating,
    SqliteLearningPackageRepository,
    SqliteQuestionBankRepository,
    SqliteQuestionProgressRepository,
    SqliteStudyAttemptRepository,
    StudyAttempt,
    StudyQuestion,
)
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

    database_path = index_directory / "metadata.sqlite3"
    repository = SqliteDocumentRepository(database_path)

    return LibraryService(
        repository=repository,
        extractor=PdfExtractor(),
        indexer=build_persistent_document_search(
            chunker,
            index_directory,
        ),
        derived_data_cleaners=(
            SqliteDocumentSummaryRepository(database_path),
            SqliteQuestionBankRepository(database_path),
            SqliteQuestionProgressRepository(database_path),
            SqliteStudyAttemptRepository(database_path),
            SqliteLearningPackageRepository(database_path),
        ),
    )


def build_library_catalog(
    index_directory: Path,
) -> LibraryCatalog:
    """Build read-only access to persistent library metadata."""

    repository = SqliteDocumentRepository(index_directory / "metadata.sqlite3")
    return LibraryCatalog(repository)


def build_document_summary_catalog(
    index_directory: Path,
) -> DocumentSummaryCatalog:
    """Build read-only access to persisted document summaries."""

    database_path = index_directory / "metadata.sqlite3"

    return DocumentSummaryCatalog(
        documents=SqliteDocumentRepository(database_path),
        summaries=SqliteDocumentSummaryRepository(database_path),
    )


def build_question_bank_catalog(
    index_directory: Path,
) -> QuestionBankCatalog:
    """Build read-only access to persisted question banks."""

    database_path = index_directory / "metadata.sqlite3"

    return QuestionBankCatalog(
        documents=SqliteDocumentRepository(database_path),
        banks=SqliteQuestionBankRepository(database_path),
    )


def build_review_service(
    index_directory: Path,
) -> ReviewService:
    """Build persistent spaced-review coordination for one library."""

    database_path = index_directory / "metadata.sqlite3"

    return ReviewService(
        banks=build_question_bank_catalog(index_directory),
        progress=SqliteQuestionProgressRepository(database_path),
        scheduler=ReviewScheduler(),
    )


def build_study_session_service(
    index_directory: Path,
) -> StudySessionService:
    """Build persistent interactive study-session coordination."""

    database_path = index_directory / "metadata.sqlite3"
    reviewer = build_review_service(index_directory)

    # The reviewer and session share the exact bank lookup so both operations
    # are guaranteed to address the same persisted question-bank identity.
    return StudySessionService(
        banks=reviewer.banks,
        reviewer=reviewer,
        attempts=SqliteStudyAttemptRepository(database_path),
        attempt_id_factory=uuid4,
        evaluator=AnswerEvaluationService(
            HuggingFaceTextGenerator(
                max_new_tokens=(DEFAULT_ANSWER_EVALUATION_MAX_NEW_TOKENS),
            )
        ),
    )


def build_learning_package_study_service(
    library_directory: Path,
) -> LearningPackageStudyService:
    """Build package-based study coordination for one personal library."""

    database_path = library_directory / "metadata.sqlite3"

    return LearningPackageStudyService(
        packages=SqliteLearningPackageRepository(database_path),
        sessions=build_study_session_service(library_directory),
    )


def build_question_bank_service(
    index_directory: Path,
    max_new_tokens: int,
) -> QuestionBankService:
    """Build grounded question generation for one persistent library."""

    database_path = index_directory / "metadata.sqlite3"
    generator = HuggingFaceTextGenerator(
        max_new_tokens=max_new_tokens,
    )

    def build_identity(
        summary: PersistedDocumentSummary,
        question_count: int,
    ) -> QuestionBankIdentity:
        return QuestionBankIdentity(
            model_name=generator.model_name,
            model_revision=generator.model_revision,
            prompt_references=(
                QUESTION_BANK_PROMPT.reference,
                QUESTION_SYSTEM_PROMPT.reference,
                QUESTION_JSON_REPAIR_PROMPT.reference,
            ),
            question_count=question_count,
            max_new_tokens=max_new_tokens,
            summary_identity_fingerprint=(summary.identity_fingerprint),
        )

    return QuestionBankService(
        summaries=build_document_summary_catalog(
            index_directory,
        ),
        generator=generator,
        banks=SqliteQuestionBankRepository(database_path),
        identity_factory=build_identity,
        max_new_tokens=max_new_tokens,
    )


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
        final_summaries=SqliteDocumentSummaryRepository(database_path),
    )


def write_learning_package_progress(phase: str) -> None:
    """Write immediate human-readable progress for product preparation."""

    messages = {
        "index": "Indexing document...",
        "summarize": "Creating document summary...",
        "questions": "Generating study questions...",
        "ready": "Learning package is ready.",
    }
    print(
        messages[phase],
        file=sys.stderr,
        flush=True,
    )


def build_learning_package_catalog(
    library_directory: Path,
) -> LearningPackageCatalog:
    """Build read-only access to user-facing learning packages."""

    repository = SqliteLearningPackageRepository(library_directory / "metadata.sqlite3")
    return LearningPackageCatalog(repository)


def _serialize_learning_package(
    package: LearningPackage,
) -> dict[str, object]:
    """Serialize one package consistently across product commands."""

    return {
        "id": str(package.id),
        "name": package.name,
        "document_id": str(package.document_id),
        "status": package.status.value,
        "summary_identity_fingerprint": (package.summary_identity_fingerprint),
        "question_bank_identity_fingerprint": (package.question_bank_identity_fingerprint),
    }


def build_learning_package_service(
    library_directory: Path,
) -> LearningPackageService:
    """Build the product workflow for one personal learning library."""

    database_path = library_directory / "metadata.sqlite3"
    packages = SqliteLearningPackageRepository(database_path)
    chunker = TextChunker(
        max_chars=DEFAULT_MAX_CHARS,
        overlap_chars=DEFAULT_OVERLAP_CHARS,
    )

    return LearningPackageService(
        packages=packages,
        documents=build_library_service(
            chunker,
            library_directory,
        ),
        summaries=build_document_summarization_service(
            library_directory,
            max_map_new_tokens=(DEFAULT_SUMMARY_MAX_MAP_NEW_TOKENS),
            max_reduce_new_tokens=(DEFAULT_SUMMARY_MAX_REDUCE_NEW_TOKENS),
            max_batch_chars=DEFAULT_SUMMARY_MAX_BATCH_CHARS,
        ),
        questions=build_question_bank_service(
            library_directory,
            max_new_tokens=DEFAULT_QUESTION_MAX_NEW_TOKENS,
        ),
        progress=write_learning_package_progress,
    )


def run_prepare(
    pdf_path: Path,
    library_directory: Path,
    name: str,
    question_count: int,
) -> int:
    """Prepare one complete learning package and emit its active state."""

    service = build_learning_package_service(library_directory)
    package = service.prepare(
        name=name,
        pdf_path=pdf_path,
        question_count=question_count,
    )

    payload = {
        "library_directory": str(library_directory),
        "package": _serialize_learning_package(package),
    }
    print(
        json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


def run_package_list(
    library_directory: Path,
) -> int:
    """List learning packages without loading model dependencies."""

    catalog = build_learning_package_catalog(library_directory)
    payload = {
        "library_directory": str(library_directory),
        "packages": [_serialize_learning_package(package) for package in catalog.list_packages()],
    }
    print(
        json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


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
    force: bool = False,
) -> int:
    """Summarize one indexed document and write the result as JSON."""

    service = build_document_summarization_service(
        index_directory,
        max_map_new_tokens,
        max_reduce_new_tokens,
        max_batch_chars,
    )
    summary = service.summarize(
        document_id,
        force=force,
    )

    payload = {
        "document_id": str(summary.document_id),
        "source": summary.source,
        "summary": summary.text,
        # Citation metadata is reconstructed from persistent chunks rather
        # than accepted from model-generated text.
        "citations": _serialize_citations(summary.citations),
        "prompts": _serialize_prompt_references(summary.prompt_references),
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


def run_question_generate(
    index_directory: Path,
    document_id: UUID,
    summary_identity_fingerprint: str,
    question_count: int,
    max_new_tokens: int,
    force: bool = False,
) -> int:
    """Generate a grounded question bank and write it as JSON."""

    service = build_question_bank_service(
        index_directory=index_directory,
        max_new_tokens=max_new_tokens,
    )
    bank = service.generate(
        document_id,
        summary_identity_fingerprint,
        question_count=question_count,
        force=force,
    )

    payload = {
        "index_directory": str(index_directory),
        "document_id": str(bank.document_id),
        "summary_identity_fingerprint": (summary_identity_fingerprint),
        "question_bank_identity_fingerprint": (bank.identity_fingerprint),
        "source": bank.source,
        "questions": _serialize_study_questions(bank.questions),
        "prompts": _serialize_prompt_references(
            bank.prompt_references,
        ),
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


def run_question_list(
    index_directory: Path,
    document_id: UUID,
) -> int:
    """Write metadata for all persisted question banks of one document."""

    catalog = build_question_bank_catalog(index_directory)
    banks = catalog.list_document_banks(document_id)

    payload = {
        "index_directory": str(index_directory),
        "document_id": str(document_id),
        "question_banks": [
            {
                "identity_fingerprint": bank.identity_fingerprint,
                "source": bank.source,
                "question_count": len(bank.questions),
                "prompts": _serialize_prompt_references(
                    bank.prompt_references,
                ),
            }
            for bank in banks
        ],
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


def run_question_show(
    index_directory: Path,
    document_id: UUID,
    identity_fingerprint: str,
) -> int:
    """Write one persisted grounded question bank as JSON."""

    catalog = build_question_bank_catalog(index_directory)
    bank = catalog.get_document_bank(
        document_id,
        identity_fingerprint,
    )

    payload = {
        "index_directory": str(index_directory),
        "document_id": str(bank.document_id),
        "identity_fingerprint": bank.identity_fingerprint,
        "source": bank.source,
        "questions": _serialize_study_questions(
            bank.questions,
        ),
        "prompts": _serialize_prompt_references(
            bank.prompt_references,
        ),
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


def _write_study_result(
    attempt: StudyAttempt,
    write_line: Callable[[str], None],
) -> None:
    """Reveal trusted learning material after a written answer."""

    evaluation = attempt.evaluation

    if evaluation is None:
        raise RuntimeError("Automatic answer evaluation returned no feedback")

    # Reveal trusted learning material only after the learner has committed to
    # a written answer and the automatic evaluation has completed.
    write_line(f"Expected answer: {attempt.expected_answer}")

    for citation in attempt.citations:
        write_line(
            f"Source {citation.number}: "
            f"{citation.source}, "
            f"page {citation.page_number}, "
            f"chunk {citation.chunk_index}"
        )

    write_line(f"Evaluation: {evaluation.verdict.value} (score: {evaluation.score:.2f})")
    write_line(f"Feedback: {evaluation.feedback}")

    for concept in evaluation.missing_concepts:
        write_line(f"Missing concept: {concept}")

    write_line(f"Scheduled as: {attempt.rating.value}")
    write_line(f"Review recorded. Next due: {attempt.resulting_progress.due_at.isoformat()}")


def run_package_study(
    library_directory: Path,
    package_name: str,
    *,
    as_of: datetime | None = None,
    read_line: Callable[[str], str] = input,
    write_line: Callable[[str], None] = print,
) -> int:
    """Run one interactive study session selected by package name."""

    study_time = as_of if as_of is not None else datetime.now(UTC)
    service = build_learning_package_study_service(library_directory)
    due = service.next_due(
        package_name,
        as_of=study_time,
    )

    if due is None:
        write_line("No study questions are due.")
        return 0

    answer_text = capture_study_answer(
        due.question,
        read_line=read_line,
        write_line=write_line,
    )
    attempt = service.record_answer(
        package_name,
        due.question.number,
        answer_text=answer_text,
        answered_at=study_time,
    )
    _write_study_result(attempt, write_line)

    return 0


def run_study(
    index_directory: Path,
    document_id: UUID,
    question_bank_identity_fingerprint: str,
    *,
    as_of: datetime | None = None,
    read_line: Callable[[str], str] = input,
    write_line: Callable[[str], None] = print,
) -> int:
    """Run one interactive due-question study session."""

    study_time = as_of if as_of is not None else datetime.now(UTC)
    service = build_study_session_service(index_directory)
    due = service.next_due(
        document_id,
        question_bank_identity_fingerprint,
        as_of=study_time,
    )

    if due is None:
        write_line("No study questions are due.")
        return 0

    answer_text = capture_study_answer(
        due.question,
        read_line=read_line,
        write_line=write_line,
    )
    attempt = service.record_answer(
        document_id,
        question_bank_identity_fingerprint,
        due.question.number,
        answer_text=answer_text,
        answered_at=study_time,
    )
    evaluation = attempt.evaluation
    if evaluation is None:
        raise RuntimeError("Automatic answer evaluation returned no feedback")

    # Reveal trusted learning material only after the learner has committed to
    # a written answer and the automatic evaluation has completed.
    write_line(f"Expected answer: {attempt.expected_answer}")
    for citation in attempt.citations:
        write_line(
            f"Source {citation.number}: "
            f"{citation.source}, "
            f"page {citation.page_number}, "
            f"chunk {citation.chunk_index}"
        )

    write_line(f"Evaluation: {evaluation.verdict.value} (score: {evaluation.score:.2f})")
    write_line(f"Feedback: {evaluation.feedback}")
    for concept in evaluation.missing_concepts:
        write_line(f"Missing concept: {concept}")

    write_line(f"Scheduled as: {attempt.rating.value}")
    write_line(f"Review recorded. Next due: {attempt.resulting_progress.due_at.isoformat()}")
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


def _serialize_study_questions(
    questions: Sequence[StudyQuestion],
) -> list[dict[str, object]]:
    """Convert grounded study questions to stable CLI JSON."""

    return [
        {
            "number": question.number,
            "text": question.text,
            "expected_answer": question.expected_answer,
            "citations": _serialize_citations(
                question.citations,
            ),
        }
        for question in questions
    ]


def _serialize_question_progress(
    progress: QuestionProgress,
) -> dict[str, object]:
    """Convert one current review schedule to stable CLI JSON."""

    return {
        "repetition_count": progress.repetition_count,
        "interval_days": progress.interval_days,
        "ease_factor": progress.ease_factor,
        "due_at": progress.due_at.isoformat(),
        "last_reviewed_at": (
            progress.last_reviewed_at.isoformat() if progress.last_reviewed_at is not None else None
        ),
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
        "citations": _serialize_citations(answer.citations),
        "prompts": _serialize_prompt_references(answer.prompt_references),
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


def _serialize_citations(
    citations: Sequence[Citation],
) -> list[dict[str, object]]:
    """Convert grounded citations to their stable CLI JSON representation."""

    return [
        {
            "number": citation.number,
            "source": citation.source,
            "page_number": citation.page_number,
            "chunk_index": citation.chunk_index,
            "excerpt": citation.excerpt,
        }
        for citation in citations
    ]


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


def run_summary_list(
    index_directory: Path,
    document_id: UUID,
) -> int:
    """Write metadata for all persisted summaries of one document."""

    catalog = build_document_summary_catalog(index_directory)
    summaries = catalog.list_document_summaries(document_id)

    payload = {
        "index_directory": str(index_directory),
        "document_id": str(document_id),
        "summaries": [
            {
                "identity_fingerprint": summary.identity_fingerprint,
                "source": summary.source,
                "citation_count": len(summary.citations),
                "prompts": _serialize_prompt_references(
                    summary.prompt_references,
                ),
            }
            for summary in summaries
        ],
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


def run_summary_show(
    index_directory: Path,
    document_id: UUID,
    identity_fingerprint: str,
) -> int:
    """Write one persisted summary with its provenance as JSON."""

    catalog = build_document_summary_catalog(index_directory)
    summary = catalog.get_document_summary(
        document_id,
        identity_fingerprint,
    )

    payload = {
        "index_directory": str(index_directory),
        "document_id": str(summary.document_id),
        "identity_fingerprint": summary.identity_fingerprint,
        "source": summary.source,
        "summary": summary.text,
        "citations": _serialize_citations(summary.citations),
        "prompts": _serialize_prompt_references(
            summary.prompt_references,
        ),
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


def run_review_due(
    index_directory: Path,
    document_id: UUID,
    question_bank_identity_fingerprint: str,
    limit: int,
    *,
    as_of: datetime | None = None,
) -> int:
    """Write due questions and their current schedules as JSON."""

    query_time = as_of if as_of is not None else datetime.now(UTC)
    service = build_review_service(index_directory)
    due_questions = service.list_due(
        document_id,
        question_bank_identity_fingerprint,
        as_of=query_time,
        limit=limit,
    )

    payload = {
        "index_directory": str(index_directory),
        "document_id": str(document_id),
        "question_bank_identity_fingerprint": (question_bank_identity_fingerprint),
        "as_of": query_time.isoformat(),
        "questions": [
            {
                "number": item.question.number,
                "text": item.question.text,
                "expected_answer": item.question.expected_answer,
                "citations": _serialize_citations(
                    item.question.citations,
                ),
                "progress": (
                    _serialize_question_progress(item.progress)
                    if item.progress is not None
                    else None
                ),
            }
            for item in due_questions
        ],
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


def run_review_record(
    index_directory: Path,
    document_id: UUID,
    question_bank_identity_fingerprint: str,
    question_number: int,
    rating: ReviewRating,
    *,
    reviewed_at: datetime | None = None,
) -> int:
    """Record one rating and write the updated schedule as JSON."""

    review_time = reviewed_at if reviewed_at is not None else datetime.now(UTC)
    service = build_review_service(index_directory)
    progress = service.record_review(
        document_id,
        question_bank_identity_fingerprint,
        question_number,
        rating,
        reviewed_at=review_time,
    )

    payload = {
        "index_directory": str(index_directory),
        "document_id": str(document_id),
        "question_bank_identity_fingerprint": (question_bank_identity_fingerprint),
        "question_number": question_number,
        "rating": rating.value,
        "progress": _serialize_question_progress(progress),
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0
