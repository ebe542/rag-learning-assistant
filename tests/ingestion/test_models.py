import pytest

from rag_learning_assistant.ingestion import Document, Page


def test_document_text_ignores_empty_pages() -> None:
    document = Document(
        source="book.pdf",
        pages=(Page(1, "", "book.pdf"), Page(2, "Content", "book.pdf")),
    )

    assert document.text == "Content"


def test_page_numbers_must_be_positive() -> None:
    with pytest.raises(ValueError, match="start at 1"):
        Page(number=0, text="text", source="book.pdf")
