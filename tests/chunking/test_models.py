from uuid import UUID

import pytest

from rag_learning_assistant.chunking import Chunk


def test_chunk_stores_text_and_source_metadata() -> None:
    chunk = Chunk(
        text="Python functions can return values.",
        source="python-book.pdf",
        page_number=12,
        index=3,
    )

    assert chunk.text == "Python functions can return values."
    assert chunk.source == "python-book.pdf"
    assert chunk.page_number == 12
    assert chunk.index == 3


def test_chunk_page_number_must_be_positive() -> None:
    with pytest.raises(ValueError, match="Page numbers start at 1"):
        Chunk(
            text="Content",
            source="book.pdf",
            page_number=0,
            index=0,
        )


def test_chunk_index_must_not_be_negative() -> None:
    with pytest.raises(ValueError, match="Chunk index must not be negative"):
        Chunk(
            text="Content",
            source="book.pdf",
            page_number=1,
            index=-1,
        )


@pytest.mark.parametrize("text", ["", "   ", "\n\t"])
def test_chunk_text_must_not_be_blank(text: str) -> None:
    with pytest.raises(ValueError, match="Chunk text must not be blank"):
        Chunk(
            text=text,
            source="book.pdf",
            page_number=1,
            index=0,
        )


@pytest.mark.parametrize("source", ["", "   ", "\n\t"])
def test_chunk_source_must_not_be_blank(source: str) -> None:
    with pytest.raises(ValueError, match="Chunk source must not be blank"):
        Chunk(
            text="Content",
            source=source,
            page_number=1,
            index=0,
        )


def test_chunk_can_reference_an_indexed_document() -> None:
    document_id = UUID("12345678-1234-5678-1234-567812345678")

    chunk = Chunk(
        text="Python functions",
        source="python-book.pdf",
        page_number=1,
        index=0,
        document_id=document_id,
    )

    assert chunk.document_id == document_id
