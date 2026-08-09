from uuid import UUID

import pytest

from rag_learning_assistant.application.library import DocumentNotFoundError
from rag_learning_assistant.application.summarization import (
    DocumentSummarizationService,
    DocumentSummary,
)
from rag_learning_assistant.chunking import Chunk
from rag_learning_assistant.generation import Citation, GenerationResult
from rag_learning_assistant.library import IndexedDocument


class RecordingDocumentLookup:
    def __init__(self, document: IndexedDocument) -> None:
        self.document = document
        self.requested_ids: list[UUID] = []

    def find_by_id(self, document_id: UUID) -> IndexedDocument | None:
        self.requested_ids.append(document_id)
        return self.document if document_id == self.document.id else None


class RecordingChunkReader:
    def __init__(self, chunks: list[Chunk]) -> None:
        self.chunks = chunks
        self.requested_ids: list[UUID] = []

    def list_document_chunks(self, document_id: UUID) -> list[Chunk]:
        self.requested_ids.append(document_id)
        return list(self.chunks)


class RecordingSummaryGenerator:
    def __init__(self, result: GenerationResult) -> None:
        self.result = result
        self.prompts: list[str] = []

    def generate(self, prompt: str) -> GenerationResult:
        self.prompts.append(prompt)
        return self.result


class SequentialSummaryGenerator:
    def __init__(self, results: list[GenerationResult]) -> None:
        self.results = list(results)
        self.prompts: list[str] = []

    def generate(self, prompt: str) -> GenerationResult:
        self.prompts.append(prompt)
        return self.results[len(self.prompts) - 1]


class MissingDocumentLookup:
    def __init__(self) -> None:
        self.requested_ids: list[UUID] = []

    def find_by_id(self, document_id: UUID) -> None:
        self.requested_ids.append(document_id)
        return None


def test_document_summary_contains_source_text_and_citations() -> None:
    document_id = UUID("12345678-1234-5678-1234-567812345678")
    citation = Citation(
        number=1,
        source="python-book.pdf",
        page_number=20,
        chunk_index=50,
        excerpt="The book teaches Python through practice and repetition.",
    )

    summary = DocumentSummary(
        document_id=document_id,
        source="python-book.pdf",
        text="The book introduces Python through repeated practical exercises.",
        citations=(citation,),
    )

    assert summary.document_id == document_id
    assert summary.source == "python-book.pdf"
    assert summary.text == ("The book introduces Python through repeated practical exercises.")
    assert summary.citations == (citation,)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("source", "", "Summary source must not be blank"),
        ("source", "   ", "Summary source must not be blank"),
        ("text", "", "Summary text must not be blank"),
        ("text", "   ", "Summary text must not be blank"),
    ],
)
def test_document_summary_rejects_blank_text_fields(
    field: str,
    value: str,
    message: str,
) -> None:
    values = {
        "document_id": UUID("12345678-1234-5678-1234-567812345678"),
        "source": "python-book.pdf",
        "text": "A grounded document summary.",
        "citations": (
            Citation(
                number=1,
                source="python-book.pdf",
                page_number=20,
                chunk_index=50,
                excerpt="Python is learned through practice.",
            ),
        ),
    }
    values[field] = value

    with pytest.raises(ValueError, match=message):
        DocumentSummary(**values)


def test_document_summary_requires_at_least_one_citation() -> None:
    with pytest.raises(
        ValueError,
        match="Summary must contain at least one citation",
    ):
        DocumentSummary(
            document_id=UUID("12345678-1234-5678-1234-567812345678"),
            source="python-book.pdf",
            text="A summary without evidence.",
            citations=(),
        )


def test_summarize_reads_complete_document_and_maps_citations() -> None:
    document_id = UUID("12345678-1234-5678-1234-567812345678")
    document = IndexedDocument(
        id=document_id,
        source="python-book.pdf",
        content_sha256="a" * 64,
        page_count=2,
        chunk_count=2,
    )
    chunks = [
        Chunk(
            text="Python programs consist of instructions.",
            source="python-book.pdf",
            page_number=1,
            index=0,
            document_id=document_id,
        ),
        Chunk(
            text="Practice and repetition build programming skills.",
            source="python-book.pdf",
            page_number=2,
            index=1,
            document_id=document_id,
        ),
    ]
    documents = RecordingDocumentLookup(document)
    chunk_reader = RecordingChunkReader(chunks)
    generator = RecordingSummaryGenerator(
        GenerationResult(
            text="The book teaches Python through practice.",
            citation_numbers=(1, 2),
        )
    )
    service = DocumentSummarizationService(
        documents=documents,
        chunks=chunk_reader,
        generator=generator,
    )

    summary = service.summarize(document_id)

    assert summary == DocumentSummary(
        document_id=document_id,
        source="python-book.pdf",
        text="The book teaches Python through practice.",
        citations=(
            Citation(
                number=1,
                source="python-book.pdf",
                page_number=1,
                chunk_index=0,
                excerpt="Python programs consist of instructions.",
            ),
            Citation(
                number=2,
                source="python-book.pdf",
                page_number=2,
                chunk_index=1,
                excerpt="Practice and repetition build programming skills.",
            ),
        ),
    )
    assert documents.requested_ids == [document_id]
    assert chunk_reader.requested_ids == [document_id]
    assert len(generator.prompts) == 1
    assert '<context number="1">' in generator.prompts[0]
    assert '<context number="2">' in generator.prompts[0]
    assert chunks[0].text in generator.prompts[0]
    assert chunks[1].text in generator.prompts[0]


def test_summarize_rejects_incomplete_chunk_storage_before_generation() -> None:
    document_id = UUID("12345678-1234-5678-1234-567812345678")
    document = IndexedDocument(
        id=document_id,
        source="python-book.pdf",
        content_sha256="a" * 64,
        page_count=2,
        chunk_count=2,
    )
    stored_chunks = [
        Chunk(
            text="Only the first chunk survived.",
            source="python-book.pdf",
            page_number=1,
            index=0,
            document_id=document_id,
        )
    ]
    documents = RecordingDocumentLookup(document)
    chunk_reader = RecordingChunkReader(stored_chunks)
    generator = RecordingSummaryGenerator(
        GenerationResult(
            text="This summary would be incomplete.",
            citation_numbers=(1,),
        )
    )
    service = DocumentSummarizationService(
        documents=documents,
        chunks=chunk_reader,
        generator=generator,
    )

    with pytest.raises(
        RuntimeError,
        match="Stored chunk count does not match document metadata",
    ):
        service.summarize(document_id)

    assert generator.prompts == []


def test_summarize_batches_long_documents_with_global_context_numbers() -> None:
    document_id = UUID("12345678-1234-5678-1234-567812345678")
    document = IndexedDocument(
        id=document_id,
        source="python-book.pdf",
        content_sha256="a" * 64,
        page_count=3,
        chunk_count=3,
    )
    chunks = [
        Chunk(
            text="AAAAAA",
            source="python-book.pdf",
            page_number=1,
            index=0,
            document_id=document_id,
        ),
        Chunk(
            text="BBBBBB",
            source="python-book.pdf",
            page_number=2,
            index=1,
            document_id=document_id,
        ),
        Chunk(
            text="CCCC",
            source="python-book.pdf",
            page_number=3,
            index=2,
            document_id=document_id,
        ),
    ]
    generator = SequentialSummaryGenerator(
        [
            GenerationResult(
                text="First section summary.",
                citation_numbers=(1,),
            ),
            GenerationResult(
                text="Second section summary.",
                citation_numbers=(2, 3),
            ),
            GenerationResult(
                text="Complete document summary.",
                citation_numbers=(1, 3),
            ),
        ]
    )
    service = DocumentSummarizationService(
        documents=RecordingDocumentLookup(document),
        chunks=RecordingChunkReader(chunks),
        generator=generator,
        max_batch_chars=10,
    )

    summary = service.summarize(document_id)

    assert summary.text == "Complete document summary."
    assert [citation.number for citation in summary.citations] == [1, 3]
    assert len(generator.prompts) == 3

    assert '<context number="1">' in generator.prompts[0]
    assert '<context number="2">' not in generator.prompts[0]
    assert '<context number="2">' in generator.prompts[1]
    assert '<context number="3">' in generator.prompts[1]

    reduction_prompt = generator.prompts[2]
    assert "First section summary." in reduction_prompt
    assert "Second section summary." in reduction_prompt
    assert "original context numbers: 1" in reduction_prompt
    assert "original context numbers: 2, 3" in reduction_prompt


def test_reduction_rejects_citation_not_supported_by_partial_summaries() -> None:
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
    generator = SequentialSummaryGenerator(
        [
            GenerationResult(
                text="First partial summary.",
                citation_numbers=(1,),
            ),
            GenerationResult(
                text="Second partial summary.",
                citation_numbers=(2,),
            ),
            GenerationResult(
                text="Combined summary.",
                citation_numbers=(3,),
            ),
        ]
    )
    service = DocumentSummarizationService(
        RecordingDocumentLookup(document),
        RecordingChunkReader(chunks),
        generator,
        max_batch_chars=15,
    )

    with pytest.raises(
        ValueError,
        match=("Citation number 3 is not supported by a section summary"),
    ):
        service.summarize(document_id)


@pytest.mark.parametrize("max_batch_chars", [0, -1])
def test_summarization_requires_positive_batch_size(
    max_batch_chars: int,
) -> None:
    document = IndexedDocument(
        id=UUID("aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa"),
        source="course.pdf",
        content_sha256="a" * 64,
        page_count=1,
        chunk_count=1,
    )

    with pytest.raises(
        ValueError,
        match="max_batch_chars must be positive",
    ):
        DocumentSummarizationService(
            RecordingDocumentLookup(document),
            RecordingChunkReader([]),
            RecordingSummaryGenerator(
                GenerationResult(
                    text="Unused.",
                    citation_numbers=(1,),
                )
            ),
            max_batch_chars=max_batch_chars,
        )


def test_summarize_rejects_unknown_document_before_reading_chunks() -> None:
    document_id = UUID("aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa")
    documents = MissingDocumentLookup()
    chunks = RecordingChunkReader([])
    generator = RecordingSummaryGenerator(
        GenerationResult(
            text="Unused.",
            citation_numbers=(1,),
        )
    )
    service = DocumentSummarizationService(
        documents,
        chunks,
        generator,
    )

    with pytest.raises(
        DocumentNotFoundError,
        match=f"Document does not exist: {document_id}",
    ):
        service.summarize(document_id)

    assert documents.requested_ids == [document_id]
    assert chunks.requested_ids == []
    assert generator.prompts == []


def test_summarize_rejects_document_without_chunks_before_generation() -> None:
    document_id = UUID("aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa")
    document = IndexedDocument(
        id=document_id,
        source="empty.pdf",
        content_sha256="a" * 64,
        page_count=1,
        chunk_count=0,
    )
    chunks = RecordingChunkReader([])
    generator = RecordingSummaryGenerator(
        GenerationResult(
            text="Unused.",
            citation_numbers=(1,),
        )
    )
    service = DocumentSummarizationService(
        RecordingDocumentLookup(document),
        chunks,
        generator,
    )

    with pytest.raises(
        ValueError,
        match="Document has no chunks to summarize",
    ):
        service.summarize(document_id)

    assert generator.prompts == []
