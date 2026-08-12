from uuid import UUID

from rag_learning_assistant.application import (
    DocumentSummarizationService,
)
from rag_learning_assistant.chunking import Chunk
from rag_learning_assistant.generation import (
    GenerationIdentity,
    GenerationResult,
    PromptTemplate,
)
from rag_learning_assistant.generation.cache import (
    CachedSummaryBatch,
)
from rag_learning_assistant.library import IndexedDocument


class StaticDocumentLookup:
    def __init__(self, document: IndexedDocument) -> None:
        self.document = document

    def find_by_id(
        self,
        document_id: UUID,
    ) -> IndexedDocument | None:
        if document_id == self.document.id:
            return self.document
        return None


class StaticChunkReader:
    def __init__(self, chunks: list[Chunk]) -> None:
        self.chunks = chunks

    def list_document_chunks(
        self,
        document_id: UUID,
    ) -> list[Chunk]:
        return list(self.chunks)


class SequentialGenerator:
    def __init__(
        self,
        results: list[GenerationResult],
    ) -> None:
        self.results = results
        self.prompts: list[str] = []

    def generate(
        self,
        prompt: str,
        *,
        max_new_tokens: int | None = None,
    ) -> GenerationResult:
        self.prompts.append(prompt)
        return self.results[len(self.prompts) - 1]


class RecordingSummaryCache:
    def __init__(
        self,
        batches: list[CachedSummaryBatch],
    ) -> None:
        self.batches = {
            (
                batch.identity_fingerprint,
                batch.batch_number,
            ): batch
            for batch in batches
        }
        self.find_calls: list[tuple[str, int]] = []
        self.saved: list[CachedSummaryBatch] = []

    def find_batch(
        self,
        identity_fingerprint: str,
        batch_number: int,
    ) -> CachedSummaryBatch | None:
        key = (identity_fingerprint, batch_number)
        self.find_calls.append(key)
        return self.batches.get(key)

    def save_batch(
        self,
        batch: CachedSummaryBatch,
    ) -> None:
        self.saved.append(batch)
        self.batches[
            (
                batch.identity_fingerprint,
                batch.batch_number,
            )
        ] = batch


def test_summarize_reuses_cached_map_batch() -> None:
    document_id = UUID("aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa")
    document = IndexedDocument(
        id=document_id,
        source="course.pdf",
        content_sha256="b" * 64,
        page_count=3,
        chunk_count=3,
    )
    chunks = [
        Chunk(
            text="AAAAAA",
            source="course.pdf",
            page_number=1,
            index=0,
            document_id=document_id,
        ),
        Chunk(
            text="BBBBBB",
            source="course.pdf",
            page_number=2,
            index=1,
            document_id=document_id,
        ),
        Chunk(
            text="CCCC",
            source="course.pdf",
            page_number=3,
            index=2,
            document_id=document_id,
        ),
    ]
    prompt = PromptTemplate(
        name="summarization.map",
        version=1,
        text="Summarize supplied contexts.",
    )
    identity = GenerationIdentity(
        model_name="Qwen/Qwen3-1.7B",
        model_revision="c" * 40,
        prompt_references=(prompt.reference,),
        max_map_new_tokens=192,
        max_reduce_new_tokens=256,
        max_batch_chars=10,
        document_content_sha256=document.content_sha256,
    )
    cached_first_batch = CachedSummaryBatch(
        identity_fingerprint=identity.fingerprint,
        batch_number=1,
        first_context_number=1,
        last_context_number=1,
        result=GenerationResult(
            text="Cached first section.",
            citation_numbers=(1,),
        ),
    )
    cache = RecordingSummaryCache([cached_first_batch])
    generator = SequentialGenerator(
        [
            GenerationResult(
                text="Generated second section.",
                citation_numbers=(2, 3),
            ),
            GenerationResult(
                text="Complete summary.",
                citation_numbers=(1, 2, 3),
            ),
        ]
    )
    service = DocumentSummarizationService(
        documents=StaticDocumentLookup(document),
        chunks=StaticChunkReader(chunks),
        generator=generator,
        max_batch_chars=10,
        max_reduce_new_tokens=256,
        cache=cache,
        identity_factory=lambda indexed_document: identity,
    )

    summary = service.summarize(document_id)

    assert summary.text == "Complete summary."
    assert len(generator.prompts) == 2
    assert "Cached first section." in generator.prompts[1]
    assert cache.find_calls == [
        (identity.fingerprint, 1),
        (identity.fingerprint, 2),
    ]
    assert len(cache.saved) == 1
    assert cache.saved[0].batch_number == 2
    assert cache.saved[0].first_context_number == 2
    assert cache.saved[0].last_context_number == 3
