from rag_learning_assistant.chunking import Chunk
from rag_learning_assistant.retrieval import SearchResult


def test_search_result_stores_chunk_and_score() -> None:
    chunk = Chunk(
        text="Python uses indentation.",
        source="python-book.pdf",
        page_number=4,
        index=2,
    )

    result = SearchResult(chunk=chunk, score=0.85)

    assert result.chunk is chunk
    assert result.score == 0.85
