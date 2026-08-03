from collections.abc import Sequence

from rag_learning_assistant.application import DocumentSearchService
from rag_learning_assistant.chunking import Chunk, TextChunker
from rag_learning_assistant.ingestion import Document, Page
from rag_learning_assistant.retrieval import SearchResult


class RecordingRetrievalService:
    """Record indexed chunks without creating real embeddings."""

    def __init__(self) -> None:
        self.indexed_chunks: list[Chunk] = []
        self.search_calls: list[tuple[str, int]] = []
        self.results: list[SearchResult] = []

    def index_chunks(self, chunks: Sequence[Chunk]) -> None:
        self.indexed_chunks.extend(chunks)

    def search(self, query: str, limit: int) -> list[SearchResult]:
        self.search_calls.append((query, limit))
        return self.results


def test_index_document_chunks_and_indexes_all_pages() -> None:
    document = Document(
        source="python-book.pdf",
        pages=(
            Page(1, "Python functions", "python-book.pdf"),
            Page(2, "Python classes", "python-book.pdf"),
        ),
    )
    retrieval = RecordingRetrievalService()
    service = DocumentSearchService(
        chunker=TextChunker(max_chars=100, overlap_chars=0),
        retrieval=retrieval,
    )

    chunks = service.index_document(document)

    assert chunks == [
        Chunk(
            text="Python functions",
            source="python-book.pdf",
            page_number=1,
            index=0,
        ),
        Chunk(
            text="Python classes",
            source="python-book.pdf",
            page_number=2,
            index=1,
        ),
    ]
    assert retrieval.indexed_chunks == chunks


def test_search_delegates_query_and_limit_to_retrieval() -> None:
    chunk = Chunk(
        text="Python functions",
        source="python-book.pdf",
        page_number=1,
        index=0,
    )
    expected = [SearchResult(chunk=chunk, score=0.9)]
    retrieval = RecordingRetrievalService()
    retrieval.results = expected
    service = DocumentSearchService(
        chunker=TextChunker(max_chars=100, overlap_chars=0),
        retrieval=retrieval,
    )

    results = service.search("How do functions work?", limit=3)

    assert results is expected
    assert retrieval.search_calls == [("How do functions work?", 3)]
