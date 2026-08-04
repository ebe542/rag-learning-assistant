from uuid import UUID

import pytest

from rag_learning_assistant.library import IndexedDocument


@pytest.mark.parametrize("source", ["", "   "])
def test_indexed_document_rejects_blank_source(source: str) -> None:
    with pytest.raises(
        ValueError,
        match="Document source must not be blank",
    ):
        IndexedDocument(
            id=UUID("12345678-1234-5678-1234-567812345678"),
            source=source,
            content_sha256="a" * 64,
            page_count=120,
            chunk_count=758,
        )


@pytest.mark.parametrize(
    "content_sha256",
    [
        "",
        "a" * 63,
        "g" * 64,
    ],
)
def test_indexed_document_rejects_invalid_content_hash(
    content_sha256: str,
) -> None:
    with pytest.raises(
        ValueError,
        match="Content hash must be a SHA-256 hexadecimal value",
    ):
        IndexedDocument(
            id=UUID("12345678-1234-5678-1234-567812345678"),
            source="python-book.pdf",
            content_sha256=content_sha256,
            page_count=120,
            chunk_count=758,
        )


@pytest.mark.parametrize("page_count", [0, -1])
def test_indexed_document_requires_positive_page_count(
    page_count: int,
) -> None:
    with pytest.raises(
        ValueError,
        match="Page count must be positive",
    ):
        IndexedDocument(
            id=UUID("12345678-1234-5678-1234-567812345678"),
            source="python-book.pdf",
            content_sha256="a" * 64,
            page_count=page_count,
            chunk_count=758,
        )


def test_indexed_document_rejects_negative_chunk_count() -> None:
    with pytest.raises(
        ValueError,
        match="Chunk count must not be negative",
    ):
        IndexedDocument(
            id=UUID("12345678-1234-5678-1234-567812345678"),
            source="python-book.pdf",
            content_sha256="a" * 64,
            page_count=120,
            chunk_count=-1,
        )
