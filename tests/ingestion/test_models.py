import pytest

from rag_learning_assistant.ingestion import (
    Document,
    Page,
    PageTextOrigin,
    has_usable_ocr_text,
)


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


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Recognized paragraph with ordinary words.", True),
        ("A", True),
        ("Short title", True),
        ("1234567890 -- ???", True),
        ("∫ 0 1 + 42", True),
        ("-- ???", False),
        ("", False),
    ],
)
def test_ocr_text_requires_a_sane_character_distribution(
    text: str,
    expected: bool,
) -> None:
    assert has_usable_ocr_text(text) is expected


@pytest.mark.parametrize(
    ("page", "expected"),
    [
        (Page(1, "Native text", "book.pdf"), PageTextOrigin.NATIVE),
        (
            Page(1, "OCR text", "book.pdf", ocr_applied=True, ocr_text_usable=True),
            PageTextOrigin.OCR,
        ),
        (
            Page(1, "42", "book.pdf", ocr_applied=True, ocr_text_usable=True),
            PageTextOrigin.OCR,
        ),
        (Page(1, "", "book.pdf"), PageTextOrigin.NONE),
        (
            Page(1, "", "book.pdf", ocr_applied=True, ocr_text_usable=False),
            PageTextOrigin.NONE,
        ),
    ],
)
def test_page_reports_final_text_origin(
    page: Page,
    expected: PageTextOrigin,
) -> None:
    assert page.text_origin is expected


def test_numeric_ocr_text_can_enter_the_processing_pipeline() -> None:
    page = Page(
        1,
        "42",
        "mathematics.pdf",
        ocr_applied=True,
        ocr_text_usable=True,
    )

    assert page.has_machine_readable_text is True


def test_page_numbers_must_be_positive() -> None:
    with pytest.raises(ValueError, match="start at 1"):
        Page(number=0, text="text", source="book.pdf")
