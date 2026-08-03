import pytest

from rag_learning_assistant.chunking import Chunk, TextChunker
from rag_learning_assistant.ingestion import Page


def test_short_page_becomes_one_chunk() -> None:
    page = Page(
        number=3,
        text="Python functions can return values.",
        source="python-book.pdf",
    )
    chunker = TextChunker(max_chars=1000, overlap_chars=150)

    chunks = chunker.chunk_pages([page])

    assert chunks == [
        Chunk(
            text="Python functions can return values.",
            source="python-book.pdf",
            page_number=3,
            index=0,
        )
    ]


def test_empty_pages_are_ignored() -> None:
    pages = [
        Page(number=1, text="   \n", source="book.pdf"),
        Page(number=2, text="Useful content", source="book.pdf"),
    ]
    chunker = TextChunker(max_chars=1000, overlap_chars=150)

    chunks = chunker.chunk_pages(pages)

    assert chunks == [
        Chunk(
            text="Useful content",
            source="book.pdf",
            page_number=2,
            index=0,
        )
    ]


def test_long_page_splits_at_word_boundaries() -> None:
    page = Page(
        number=1,
        text="one two three four",
        source="book.pdf",
    )
    chunker = TextChunker(max_chars=9, overlap_chars=0)

    chunks = chunker.chunk_pages([page])

    assert [chunk.text for chunk in chunks] == [
        "one two",
        "three",
        "four",
    ]
    assert [chunk.index for chunk in chunks] == [0, 1, 2]
    assert all(chunk.source == "book.pdf" for chunk in chunks)
    assert all(chunk.page_number == 1 for chunk in chunks)


@pytest.mark.parametrize("max_chars", [0, -1])
def test_max_chars_must_be_positive(max_chars: int) -> None:
    with pytest.raises(ValueError, match="max_chars must be positive"):
        TextChunker(max_chars=max_chars, overlap_chars=0)


def test_overlap_chars_must_not_be_negative() -> None:
    with pytest.raises(ValueError, match="overlap_chars must not be negative"):
        TextChunker(max_chars=100, overlap_chars=-1)


@pytest.mark.parametrize("overlap_chars", [100, 101])
def test_overlap_must_be_smaller_than_max_chars(overlap_chars: int) -> None:
    with pytest.raises(ValueError, match="overlap_chars must be smaller than max_chars"):
        TextChunker(max_chars=100, overlap_chars=overlap_chars)


def test_chunks_overlap_at_word_boundaries() -> None:
    page = Page(
        number=1,
        text="one two three four five",
        source="book.pdf",
    )
    chunker = TextChunker(max_chars=13, overlap_chars=5)

    chunks = chunker.chunk_pages([page])

    assert [chunk.text for chunk in chunks] == [
        "one two three",
        "three four",
        "four five",
    ]


def test_word_longer_than_max_chars_is_split() -> None:
    page = Page(
        number=1,
        text="abcdefgh",
        source="book.pdf",
    )
    chunker = TextChunker(max_chars=5, overlap_chars=0)

    chunks = chunker.chunk_pages([page])

    assert [chunk.text for chunk in chunks] == ["abcde", "fgh"]
    assert all(len(chunk.text) <= 5 for chunk in chunks)
