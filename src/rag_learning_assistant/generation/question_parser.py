import json
from dataclasses import dataclass

from rag_learning_assistant.generation.prompts import PromptReference


@dataclass(frozen=True, slots=True)
class GeneratedQuestionDraft:
    """Hold one parsed model proposal before citations are resolved."""

    number: int
    text: str
    expected_answer: str
    citation_numbers: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class QuestionGenerationResult:
    """Hold parsed question drafts and the prompts used to produce them."""

    questions: tuple[GeneratedQuestionDraft, ...]
    prompt_references: tuple[PromptReference, ...]

    def __post_init__(self) -> None:
        if not self.questions:
            raise ValueError(
                "Question generation result requires at least one question",
            )

        if not self.prompt_references:
            raise ValueError(
                "Question generation result requires at least one prompt reference",
            )


def _parse_question(
    item: object,
    number: int,
) -> GeneratedQuestionDraft:
    if not isinstance(item, dict):
        raise ValueError(
            "Each generated question must be a JSON object",
        )

    expected_keys = {
        "text",
        "expected_answer",
        "citation_numbers",
    }
    if set(item) != expected_keys:
        raise ValueError(
            "Generated question must contain exactly text, expected_answer, and citation_numbers",
        )

    text = item["text"]
    if not isinstance(text, str):
        raise ValueError(
            "Generated question text must be a string",
        )
    text = text.strip()
    if not text:
        raise ValueError(
            "Generated question text must not be blank",
        )

    expected_answer = item["expected_answer"]
    if not isinstance(expected_answer, str):
        raise ValueError(
            "Generated question expected_answer must be a string",
        )
    expected_answer = expected_answer.strip()
    if not expected_answer:
        raise ValueError(
            "Generated question expected_answer must not be blank",
        )

    citation_numbers = item["citation_numbers"]
    if not isinstance(citation_numbers, list):
        raise ValueError(
            "Generated question citation_numbers must be an array",
        )

    if not citation_numbers:
        raise ValueError(
            "Generated question requires at least one citation number",
        )

    # bool is a subclass of int in Python, but JSON booleans are not valid
    # context references.
    if any(type(value) is not int for value in citation_numbers):
        raise ValueError(
            "Generated question citation_numbers must contain integers",
        )

    if any(value < 1 for value in citation_numbers):
        raise ValueError(
            "Generated question citation numbers must be positive",
        )

    if len(set(citation_numbers)) != len(citation_numbers):
        raise ValueError(
            "Generated question citation numbers must be unique",
        )

    return GeneratedQuestionDraft(
        number=number,
        text=text,
        expected_answer=expected_answer,
        citation_numbers=tuple(citation_numbers),
    )


def parse_question_generation_response(
    response: str,
) -> tuple[GeneratedQuestionDraft, ...]:
    """Parse structured question proposals from a model response."""

    try:
        payload = json.loads(response)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"Question response must be valid JSON: {exc.msg}",
        ) from exc

    if not isinstance(payload, dict):
        raise ValueError(
            "Question response must be a JSON object",
        )

    if set(payload) != {"questions"}:
        raise ValueError(
            "Question response must contain exactly the key 'questions'",
        )

    questions = payload["questions"]

    if not isinstance(questions, list):
        raise ValueError(
            "Question response questions must be an array",
        )

    if not questions:
        raise ValueError(
            "Question response requires at least one question",
        )

    drafts = tuple(
        _parse_question(item, number)
        for number, item in enumerate(
            questions,
            start=1,
        )
    )

    normalized_questions = [draft.text.casefold() for draft in drafts]
    if len(set(normalized_questions)) != len(normalized_questions):
        raise ValueError(
            "Generated question texts must be unique",
        )

    return drafts
