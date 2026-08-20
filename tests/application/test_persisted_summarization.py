from uuid import UUID

import pytest

from rag_learning_assistant.application import DocumentSummarizationService
from rag_learning_assistant.chunking import Chunk
from rag_learning_assistant.generation import (
    Citation,
    GenerationIdentity,
    GenerationResult,
    PersistedDocumentSummary,
    PromptReference,
    SqliteDocumentSummaryRepository,
)
from rag_learning_assistant.generation.cache import CachedSummaryBatch
from rag_learning_assistant.library import IndexedDocument


class RecordingMapCache:
    def __init__(
        self,
        batch: CachedSummaryBatch,
    ) -> None:
        self.batch = batch
        self.find_calls: list[tuple[str, int]] = []
        self.saved: list[CachedSummaryBatch] = []

    def find_batch(
        self,
        identity_fingerprint: str,
        batch_number: int,
    ) -> CachedSummaryBatch | None:
        self.find_calls.append((identity_fingerprint, batch_number))
        return self.batch

    def save_batch(self, batch: CachedSummaryBatch) -> None:
        self.saved.append(batch)


class InvalidCitationGenerator:
    def generate(
        self,
        prompt: str,
        *,
        max_new_tokens: int | None = None,
    ) -> GenerationResult:
        return GenerationResult(
            text="Invalid generated summary.",
            citation_numbers=(2,),
        )


def test_failed_summarization_does_not_persist_final_summary() -> None:
    document_id = UUID("aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa")
    document = IndexedDocument(
        id=document_id,
        source="document.pdf",
        content_sha256="c" * 64,
        page_count=1,
        chunk_count=1,
    )
    identity = GenerationIdentity(
        model_name="Qwen/Qwen3-1.7B",
        model_revision="b" * 40,
        prompt_references=(
            PromptReference(
                name="summarization.map",
                version=2,
                fingerprint="a" * 64,
            ),
        ),
        max_map_new_tokens=192,
        max_reduce_new_tokens=384,
        max_batch_chars=8000,
        document_content_sha256=document.content_sha256,
    )
    repository = RecordingFinalSummaryRepository()
    service = DocumentSummarizationService(
        documents=StaticDocumentLookup(document),
        chunks=StaticChunkReader(
            [
                Chunk(
                    text="Only available passage.",
                    source="document.pdf",
                    page_number=1,
                    index=0,
                    document_id=document_id,
                )
            ]
        ),
        generator=InvalidCitationGenerator(),
        max_batch_chars=8000,
        final_summaries=repository,
        identity_factory=lambda indexed_document: identity,
    )

    with pytest.raises(
        ValueError,
        match="Citation number 2 does not belong to its summary batch",
    ):
        service.summarize(document_id)

    assert repository.saved == []


class RecordingFinalSummaryRepository:
    def __init__(
        self,
        summary: PersistedDocumentSummary | None = None,
    ) -> None:
        self.summary = summary
        self.saved: list[PersistedDocumentSummary] = []
        self.replaced: list[PersistedDocumentSummary] = []

    def find(
        self,
        document_id: UUID,
        identity_fingerprint: str,
    ) -> PersistedDocumentSummary | None:
        return self.summary

    def save(self, summary: PersistedDocumentSummary) -> None:
        self.saved.append(summary)

    def replace(self, summary: PersistedDocumentSummary) -> None:
        self.replaced.append(summary)


class StaticChunkReader:
    def __init__(self, chunks: list[Chunk]) -> None:
        self.chunks = chunks

    def list_document_chunks(
        self,
        document_id: UUID,
    ) -> list[Chunk]:
        return self.chunks


class StaticGenerator:
    def __init__(
        self,
        result: GenerationResult,
    ) -> None:
        self.result = result
        self.calls: list[tuple[str, int | None]] = []

    def generate(
        self,
        prompt: str,
        *,
        max_new_tokens: int | None = None,
    ) -> GenerationResult:
        self.calls.append((prompt, max_new_tokens))
        return self.result


class StaticDocumentLookup:
    def __init__(self, document: IndexedDocument) -> None:
        self.document = document
        self.calls: list[UUID] = []

    def find_by_id(
        self,
        document_id: UUID,
    ) -> IndexedDocument | None:
        self.calls.append(document_id)
        return self.document


class StaticFinalSummaryRepository:
    def __init__(
        self,
        summary: PersistedDocumentSummary | None,
    ) -> None:
        self.summary = summary
        self.find_calls: list[tuple[UUID, str]] = []
        self.replaced: list[PersistedDocumentSummary] = []

    def find(
        self,
        document_id: UUID,
        identity_fingerprint: str,
    ) -> PersistedDocumentSummary | None:
        self.find_calls.append((document_id, identity_fingerprint))
        return self.summary

    def save(self, summary: PersistedDocumentSummary) -> None:
        raise AssertionError("Cached final summary must not be written again")

    def replace(self, summary: PersistedDocumentSummary) -> None:
        raise AssertionError("Cached final summary must not be replaced")


class FailingDocumentLookup:
    def find_by_id(self, document_id: UUID) -> None:
        raise AssertionError("Document metadata must not be loaded for a cached final summary")


class FailingChunkReader:
    def list_document_chunks(
        self,
        document_id: UUID,
    ) -> list[Chunk]:
        raise AssertionError("Chunks must not be loaded for a cached final summary")


class FailingGenerator:
    def generate(
        self,
        prompt: str,
        *,
        max_new_tokens: int | None = None,
    ) -> GenerationResult:
        raise AssertionError("Generator must not run for a cached final summary")


def test_summarize_returns_persisted_final_summary_without_generation() -> None:
    document_id = UUID("aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa")
    prompt_reference = PromptReference(
        name="summarization.reduce",
        version=4,
        fingerprint="a" * 64,
    )
    document = IndexedDocument(
        id=document_id,
        source="document.pdf",
        content_sha256="c" * 64,
        page_count=10,
        chunk_count=20,
    )
    document_lookup = StaticDocumentLookup(document)
    identity = GenerationIdentity(
        model_name="Qwen/Qwen3-1.7B",
        model_revision="b" * 40,
        prompt_references=(prompt_reference,),
        max_map_new_tokens=192,
        max_reduce_new_tokens=384,
        max_batch_chars=8000,
        document_content_sha256="c" * 64,
    )
    persisted = PersistedDocumentSummary(
        document_id=document_id,
        identity_fingerprint=identity.fingerprint,
        source="document.pdf",
        text="Previously generated summary.",
        citations=(
            Citation(
                number=1,
                source="document.pdf",
                page_number=1,
                chunk_index=0,
                excerpt="Supporting passage.",
            ),
        ),
        prompt_references=(prompt_reference,),
    )
    repository = StaticFinalSummaryRepository(persisted)
    service = DocumentSummarizationService(
        documents=document_lookup,
        chunks=FailingChunkReader(),
        generator=FailingGenerator(),
        final_summaries=repository,
        identity_factory=lambda indexed_document: identity,
    )

    summary = service.summarize(document_id)

    assert summary.document_id == document_id
    assert summary.source == "document.pdf"
    assert summary.text == "Previously generated summary."
    assert summary.citations == persisted.citations
    assert summary.prompt_references == persisted.prompt_references
    assert repository.find_calls == [
        (document_id, identity.fingerprint),
    ]
    assert document_lookup.calls == [document_id]


def test_summarize_persists_new_final_summary() -> None:
    document_id = UUID("aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa")
    document = IndexedDocument(
        id=document_id,
        source="document.pdf",
        content_sha256="c" * 64,
        page_count=1,
        chunk_count=1,
    )
    identity = GenerationIdentity(
        model_name="Qwen/Qwen3-1.7B",
        model_revision="b" * 40,
        prompt_references=(
            PromptReference(
                name="summarization.map",
                version=2,
                fingerprint="a" * 64,
            ),
        ),
        max_map_new_tokens=192,
        max_reduce_new_tokens=384,
        max_batch_chars=8000,
        document_content_sha256=document.content_sha256,
    )
    repository = RecordingFinalSummaryRepository()
    generator = StaticGenerator(
        GenerationResult(
            text="Generated final summary.",
            citation_numbers=(1,),
            prompt_references=identity.prompt_references,
        )
    )
    service = DocumentSummarizationService(
        documents=StaticDocumentLookup(document),
        chunks=StaticChunkReader(
            [
                Chunk(
                    text="Supporting passage.",
                    source="document.pdf",
                    page_number=1,
                    index=0,
                    document_id=document_id,
                )
            ]
        ),
        generator=generator,
        max_batch_chars=8000,
        final_summaries=repository,
        identity_factory=lambda indexed_document: identity,
    )

    summary = service.summarize(document_id)

    assert summary.text == "Generated final summary."
    assert repository.saved == [
        PersistedDocumentSummary(
            document_id=document_id,
            identity_fingerprint=identity.fingerprint,
            source=summary.source,
            text=summary.text,
            citations=summary.citations,
            prompt_references=summary.prompt_references,
        )
    ]


def test_persisted_final_summary_survives_service_recreation(
    tmp_path,
) -> None:
    document_id = UUID("aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa")
    document = IndexedDocument(
        id=document_id,
        source="document.pdf",
        content_sha256="c" * 64,
        page_count=1,
        chunk_count=1,
    )
    identity = GenerationIdentity(
        model_name="Qwen/Qwen3-1.7B",
        model_revision="b" * 40,
        prompt_references=(
            PromptReference(
                name="summarization.map",
                version=2,
                fingerprint="a" * 64,
            ),
        ),
        max_map_new_tokens=192,
        max_reduce_new_tokens=384,
        max_batch_chars=8000,
        document_content_sha256=document.content_sha256,
    )
    database_path = tmp_path / "metadata.sqlite3"
    first_repository = SqliteDocumentSummaryRepository(database_path)
    first_service = DocumentSummarizationService(
        documents=StaticDocumentLookup(document),
        chunks=StaticChunkReader(
            [
                Chunk(
                    text="Supporting passage.",
                    source="document.pdf",
                    page_number=1,
                    index=0,
                    document_id=document_id,
                )
            ]
        ),
        generator=StaticGenerator(
            GenerationResult(
                text="Persisted final summary.",
                citation_numbers=(1,),
                prompt_references=identity.prompt_references,
            )
        ),
        max_batch_chars=8000,
        final_summaries=first_repository,
        identity_factory=lambda indexed_document: identity,
    )

    generated = first_service.summarize(document_id)

    reopened_repository = SqliteDocumentSummaryRepository(database_path)
    reopened_service = DocumentSummarizationService(
        documents=StaticDocumentLookup(document),
        chunks=FailingChunkReader(),
        generator=FailingGenerator(),
        max_batch_chars=8000,
        final_summaries=reopened_repository,
        identity_factory=lambda indexed_document: identity,
    )

    loaded = reopened_service.summarize(document_id)

    assert loaded == generated


def test_different_generation_identity_does_not_reuse_final_summary(
    tmp_path,
) -> None:
    document_id = UUID("aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa")
    document = IndexedDocument(
        id=document_id,
        source="document.pdf",
        content_sha256="c" * 64,
        page_count=1,
        chunk_count=1,
    )
    old_identity = GenerationIdentity(
        model_name="Qwen/Qwen3-1.7B",
        model_revision="b" * 40,
        prompt_references=(
            PromptReference(
                name="summarization.map",
                version=2,
                fingerprint="a" * 64,
            ),
        ),
        max_map_new_tokens=192,
        max_reduce_new_tokens=384,
        max_batch_chars=8000,
        document_content_sha256=document.content_sha256,
    )
    new_identity = GenerationIdentity(
        model_name=old_identity.model_name,
        model_revision=old_identity.model_revision,
        prompt_references=old_identity.prompt_references,
        max_map_new_tokens=256,
        max_reduce_new_tokens=384,
        max_batch_chars=8000,
        document_content_sha256=document.content_sha256,
    )
    repository = SqliteDocumentSummaryRepository(tmp_path / "metadata.sqlite3")
    repository.save(
        PersistedDocumentSummary(
            document_id=document_id,
            identity_fingerprint=old_identity.fingerprint,
            source=document.source,
            text="Old summary.",
            citations=(
                Citation(
                    number=1,
                    source=document.source,
                    page_number=1,
                    chunk_index=0,
                    excerpt="Supporting passage.",
                ),
            ),
            prompt_references=old_identity.prompt_references,
        )
    )
    generator = StaticGenerator(
        GenerationResult(
            text="New summary.",
            citation_numbers=(1,),
            prompt_references=new_identity.prompt_references,
        )
    )
    service = DocumentSummarizationService(
        documents=StaticDocumentLookup(document),
        chunks=StaticChunkReader(
            [
                Chunk(
                    text="Supporting passage.",
                    source=document.source,
                    page_number=1,
                    index=0,
                    document_id=document_id,
                )
            ]
        ),
        generator=generator,
        max_batch_chars=8000,
        max_map_new_tokens=256,
        final_summaries=repository,
        identity_factory=lambda indexed_document: new_identity,
    )

    summary = service.summarize(document_id)

    assert summary.text == "New summary."
    assert len(generator.calls) == 1
    assert (
        repository.find(
            document_id,
            old_identity.fingerprint,
        )
        is not None
    )
    assert (
        repository.find(
            document_id,
            new_identity.fingerprint,
        )
        is not None
    )


def test_force_regenerates_and_replaces_existing_final_summary() -> None:
    document_id = UUID("aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa")
    document = IndexedDocument(
        id=document_id,
        source="document.pdf",
        content_sha256="c" * 64,
        page_count=1,
        chunk_count=1,
    )
    identity = GenerationIdentity(
        model_name="Qwen/Qwen3-1.7B",
        model_revision="b" * 40,
        prompt_references=(
            PromptReference(
                name="summarization.map",
                version=2,
                fingerprint="a" * 64,
            ),
        ),
        max_map_new_tokens=192,
        max_reduce_new_tokens=384,
        max_batch_chars=8000,
        document_content_sha256=document.content_sha256,
    )
    existing = PersistedDocumentSummary(
        document_id=document_id,
        identity_fingerprint=identity.fingerprint,
        source=document.source,
        text="Existing summary.",
        citations=(
            Citation(
                number=1,
                source=document.source,
                page_number=1,
                chunk_index=0,
                excerpt="Supporting passage.",
            ),
        ),
        prompt_references=identity.prompt_references,
    )
    repository = RecordingFinalSummaryRepository(existing)
    repository.summary = existing
    generator = StaticGenerator(
        GenerationResult(
            text="Regenerated summary.",
            citation_numbers=(1,),
            prompt_references=identity.prompt_references,
        )
    )
    service = DocumentSummarizationService(
        documents=StaticDocumentLookup(document),
        chunks=StaticChunkReader(
            [
                Chunk(
                    text="Supporting passage.",
                    source=document.source,
                    page_number=1,
                    index=0,
                    document_id=document_id,
                )
            ]
        ),
        generator=generator,
        max_batch_chars=8000,
        final_summaries=repository,
        identity_factory=lambda indexed_document: identity,
    )

    summary = service.summarize(document_id, force=True)

    assert summary.text == "Regenerated summary."
    assert len(generator.calls) == 1
    assert repository.saved == []
    assert repository.replaced == [
        PersistedDocumentSummary(
            document_id=document_id,
            identity_fingerprint=identity.fingerprint,
            source=summary.source,
            text=summary.text,
            citations=summary.citations,
            prompt_references=summary.prompt_references,
        )
    ]


def test_force_bypasses_map_cache_and_regenerates_all_batches() -> None:
    document_id = UUID("aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa")
    document = IndexedDocument(
        id=document_id,
        source="document.pdf",
        content_sha256="c" * 64,
        page_count=1,
        chunk_count=1,
    )
    identity = GenerationIdentity(
        model_name="Qwen/Qwen3-1.7B",
        model_revision="b" * 40,
        prompt_references=(
            PromptReference(
                name="summarization.map",
                version=2,
                fingerprint="a" * 64,
            ),
        ),
        max_map_new_tokens=192,
        max_reduce_new_tokens=384,
        max_batch_chars=8000,
        document_content_sha256=document.content_sha256,
    )
    map_cache = RecordingMapCache(
        CachedSummaryBatch(
            identity_fingerprint=identity.fingerprint,
            batch_number=1,
            first_context_number=1,
            last_context_number=1,
            result=GenerationResult(
                text="Cached partial summary.",
                citation_numbers=(1,),
            ),
        )
    )
    final_repository = RecordingFinalSummaryRepository()
    generator = StaticGenerator(
        GenerationResult(
            text="Freshly generated summary.",
            citation_numbers=(1,),
            prompt_references=identity.prompt_references,
        )
    )
    service = DocumentSummarizationService(
        documents=StaticDocumentLookup(document),
        chunks=StaticChunkReader(
            [
                Chunk(
                    text="Supporting passage.",
                    source=document.source,
                    page_number=1,
                    index=0,
                    document_id=document_id,
                )
            ]
        ),
        generator=generator,
        max_batch_chars=8000,
        cache=map_cache,
        final_summaries=final_repository,
        identity_factory=lambda indexed_document: identity,
    )

    summary = service.summarize(document_id, force=True)

    assert summary.text == "Freshly generated summary."
    assert len(generator.calls) == 1
    assert map_cache.find_calls == []
    assert map_cache.saved == []
    assert len(final_repository.replaced) == 1


def test_prepare_summary_returns_persisted_identity() -> None:
    document_id = UUID("aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa")
    document = IndexedDocument(
        id=document_id,
        source="document.pdf",
        content_sha256="c" * 64,
        page_count=1,
        chunk_count=1,
    )
    identity = GenerationIdentity(
        model_name="Qwen/Qwen3-1.7B",
        model_revision="b" * 40,
        prompt_references=(
            PromptReference(
                name="summarization.map",
                version=2,
                fingerprint="a" * 64,
            ),
        ),
        max_map_new_tokens=192,
        max_reduce_new_tokens=384,
        max_batch_chars=8000,
        document_content_sha256=document.content_sha256,
    )
    repository = RecordingFinalSummaryRepository()
    service = DocumentSummarizationService(
        documents=StaticDocumentLookup(document),
        chunks=StaticChunkReader(
            [
                Chunk(
                    text="Grounded source passage.",
                    source=document.source,
                    page_number=1,
                    index=0,
                    document_id=document_id,
                )
            ]
        ),
        generator=StaticGenerator(
            GenerationResult(
                text="Grounded summary.",
                citation_numbers=(1,),
            )
        ),
        identity_factory=lambda indexed_document: identity,
        final_summaries=repository,
    )

    fingerprint = service.prepare_summary(document_id)

    assert fingerprint == identity.fingerprint
    assert len(repository.saved) == 1
    assert repository.saved[0].identity_fingerprint == identity.fingerprint
