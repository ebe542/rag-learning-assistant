import pytest

from rag_learning_assistant.generation import (
    Citation,
    GenerationResult,
    GroundedAnswer,
)


def test_grounded_answer_contains_question_text_and_sources() -> None:
    citation = Citation(
        number=1,
        source="python-book.pdf",
        page_number=42,
        chunk_index=7,
        excerpt="Functions group reusable instructions.",
    )

    answer = GroundedAnswer(
        question="What is a Python function?",
        text="A function groups reusable instructions.",
        citations=(citation,),
    )

    assert answer.question == "What is a Python function?"
    assert answer.text == "A function groups reusable instructions."
    assert answer.citations == (citation,)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("source", "", "Citation source must not be blank"),
        ("source", "   ", "Citation source must not be blank"),
        ("excerpt", "", "Citation excerpt must not be blank"),
        ("excerpt", "   ", "Citation excerpt must not be blank"),
    ],
)
def test_citation_rejects_blank_text_fields(
    field: str,
    value: str,
    message: str,
) -> None:
    values = {
        "number": 1,
        "source": "python-book.pdf",
        "page_number": 1,
        "chunk_index": 0,
        "excerpt": "Python functions",
    }
    values[field] = value

    with pytest.raises(ValueError, match=message):
        Citation(**values)


@pytest.mark.parametrize("page_number", [0, -1])
def test_citation_requires_positive_page_number(
    page_number: int,
) -> None:
    with pytest.raises(
        ValueError,
        match="Citation page number must be positive",
    ):
        Citation(
            number=1,
            source="python-book.pdf",
            page_number=page_number,
            chunk_index=0,
            excerpt="Python functions",
        )


def test_citation_rejects_negative_chunk_index() -> None:
    with pytest.raises(
        ValueError,
        match="Citation chunk index must not be negative",
    ):
        Citation(
            number=1,
            source="python-book.pdf",
            page_number=1,
            chunk_index=-1,
            excerpt="Python functions",
        )


@pytest.mark.parametrize(
    ("field", "message"),
    [
        ("question", "Answer question must not be blank"),
        ("text", "Answer text must not be blank"),
    ],
)
def test_grounded_answer_rejects_blank_text(
    field: str,
    message: str,
) -> None:
    values = {
        "question": "What is Python?",
        "text": "Python is a programming language.",
        "citations": (),
    }
    values[field] = "   "

    with pytest.raises(ValueError, match=message):
        GroundedAnswer(**values)


def test_generation_result_identifies_used_contexts() -> None:
    result = GenerationResult(
        text="A function groups reusable instructions.",
        citation_numbers=(1, 3),
    )

    assert result.text == "A function groups reusable instructions."
    assert result.citation_numbers == (1, 3)


def test_generation_result_rejects_blank_text() -> None:
    with pytest.raises(
        ValueError,
        match="Generated text must not be blank",
    ):
        GenerationResult(
            text="   ",
            citation_numbers=(),
        )


@pytest.mark.parametrize("citation_number", [0, -1])
def test_generation_result_requires_positive_citation_numbers(
    citation_number: int,
) -> None:
    with pytest.raises(
        ValueError,
        match="Citation numbers must be positive",
    ):
        GenerationResult(
            text="Generated answer",
            citation_numbers=(citation_number,),
        )


def test_generation_result_rejects_duplicate_citation_numbers() -> None:
    with pytest.raises(
        ValueError,
        match="Citation numbers must be unique",
    ):
        GenerationResult(
            text="Generated answer",
            citation_numbers=(1, 1),
        )


def test_citation_requires_positive_number() -> None:
    with pytest.raises(
        ValueError,
        match="Citation number must be positive",
    ):
        Citation(
            number=0,
            source="python-book.pdf",
            page_number=42,
            chunk_index=15,
            excerpt="A class defines objects.",
        )
