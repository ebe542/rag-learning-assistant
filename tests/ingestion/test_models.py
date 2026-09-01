import pytest

from rag_learning_assistant.ingestion import Document, Page


def test_document_text_ignores_empty_pages() -> None:
    document = Document(
        source="book.pdf",
        pages=(Page(1, "", "book.pdf"), Page(2, "Content", "book.pdf")),
    )

    assert document.text == "Content"
    assert document.pages_without_machine_readable_text == (1,)


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Readable text", True),
        ("Überblick", True),
        ("E-Mail", True),
        ("A 123 --", True),
        ("123 --", False),
        ("", False),
    ],
)
def test_page_reports_machine_readable_text(text: str, expected: bool) -> None:
    page = Page(number=1, text=text, source="book.pdf")

    assert page.has_machine_readable_text is expected


def test_page_numbers_must_be_positive() -> None:
    with pytest.raises(ValueError, match="start at 1"):
        Page(number=0, text="text", source="book.pdf")
