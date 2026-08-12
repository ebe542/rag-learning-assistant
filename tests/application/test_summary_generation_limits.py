"""Tests for phase-specific document-summary generation limits."""

from uuid import UUID

from rag_learning_assistant.application import DocumentSummarizationService
from rag_learning_assistant.application.summarization import SUMMARY_MAP_PROMPT
from rag_learning_assistant.chunking import Chunk
from rag_learning_assistant.generation import GenerationResult
from rag_learning_assistant.library import IndexedDocument


class StaticDocumentLookup:
    def __init__(self, document: IndexedDocument) -> None:
        self.document = document

    def find_by_id(self, document_id: UUID) -> IndexedDocument | None:
        if document_id == self.document.id:
            return self.document

        return None


class StaticChunkReader:
    def __init__(self, chunks: list[Chunk]) -> None:
        self.chunks = chunks

    def list_document_chunks(self, document_id: UUID) -> list[Chunk]:
        return list(self.chunks)


class RecordingConfigurableGenerator:
    def __init__(self) -> None:
        self.calls: list[tuple[str, int | None]] = []

    def generate(
        self,
        prompt: str,
        *,
        max_new_tokens: int | None = None,
    ) -> GenerationResult:
        self.calls.append((prompt, max_new_tokens))

        if len(self.calls) == 1:
            return GenerationResult(
                text="First partial summary.",
                citation_numbers=(1,),
            )

        if len(self.calls) == 2:
            return GenerationResult(
                text="Second partial summary.",
                citation_numbers=(2,),
            )

        return GenerationResult(
            text="Complete document summary.",
            citation_numbers=(1, 2),
        )


def test_summarization_uses_separate_map_and_reduce_token_limits() -> None:
    document_id = UUID("aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa")
    document = IndexedDocument(
        id=document_id,
        source="course.pdf",
        content_sha256="a" * 64,
        page_count=2,
        chunk_count=2,
    )
    chunks = [
        Chunk(
            text="First section.",
            source="course.pdf",
            page_number=1,
            index=0,
            document_id=document_id,
        ),
        Chunk(
            text="Second section.",
            source="course.pdf",
            page_number=2,
            index=1,
            document_id=document_id,
        ),
    ]
    generator = RecordingConfigurableGenerator()
    service = DocumentSummarizationService(
        documents=StaticDocumentLookup(document),
        chunks=StaticChunkReader(chunks),
        generator=generator,
        max_batch_chars=15,
        max_map_new_tokens=128,
        max_reduce_new_tokens=256,
    )

    summary = service.summarize(document_id)

    assert summary.text == "Complete document summary."
    assert len(generator.calls) == 3
    assert [max_new_tokens for _, max_new_tokens in generator.calls] == [
        128,
        128,
        256,
    ]


def test_map_prompt_requires_a_concise_partial_summary() -> None:
    assert SUMMARY_MAP_PROMPT.version == 2
    assert "Use at most 80 words." in SUMMARY_MAP_PROMPT.text
    assert "Include only the most important supported claims." in SUMMARY_MAP_PROMPT.text
