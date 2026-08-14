import json

import pytest

from rag_learning_assistant.generation import (
    GeneratedQuestionDraft,
    PromptReference,
    QuestionGenerationResult,
    parse_question_generation_response,
)


def test_parse_question_generation_response_returns_numbered_drafts() -> None:
    response = """
    {
      "questions": [
        {
          "text": "What is an embedding?",
          "expected_answer": "A numeric representation of text.",
          "citation_numbers": [1, 2]
        },
        {
          "text": "Why are embeddings normalized?",
          "expected_answer": "To support comparable similarity scores.",
          "citation_numbers": [3]
        }
      ]
    }
    """

    drafts = parse_question_generation_response(response)

    assert drafts == (
        GeneratedQuestionDraft(
            number=1,
            text="What is an embedding?",
            expected_answer="A numeric representation of text.",
            citation_numbers=(1, 2),
        ),
        GeneratedQuestionDraft(
            number=2,
            text="Why are embeddings normalized?",
            expected_answer=("To support comparable similarity scores."),
            citation_numbers=(3,),
        ),
    )


@pytest.mark.parametrize(
    ("response", "message"),
    [
        (
            "not JSON",
            "Question response must be valid JSON",
        ),
        (
            "[]",
            "Question response must be a JSON object",
        ),
        (
            "{}",
            "Question response must contain exactly",
        ),
        (
            '{"questions": [], "unexpected": true}',
            "Question response must contain exactly",
        ),
        (
            '{"questions": "not an array"}',
            "Question response questions must be an array",
        ),
        (
            '{"questions": []}',
            "Question response requires at least one question",
        ),
    ],
)
def test_parse_question_generation_response_rejects_invalid_envelope(
    response: str,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        parse_question_generation_response(response)


@pytest.mark.parametrize(
    ("question_json", "message"),
    [
        (
            "42",
            "Each generated question must be a JSON object",
        ),
        (
            '{"text": "Question", "expected_answer": "Answer"}',
            "Generated question must contain exactly",
        ),
        (
            (
                '{"text": "Question", "expected_answer": "Answer", '
                '"citation_numbers": [1], "unexpected": true}'
            ),
            "Generated question must contain exactly",
        ),
        (
            ('{"text": 42, "expected_answer": "Answer", "citation_numbers": [1]}'),
            "Generated question text must be a string",
        ),
        (
            ('{"text": "Question", "expected_answer": 42, "citation_numbers": [1]}'),
            "Generated question expected_answer must be a string",
        ),
        (
            ('{"text": "Question", "expected_answer": "Answer", "citation_numbers": "1"}'),
            "Generated question citation_numbers must be an array",
        ),
        (
            ('{"text": "Question", "expected_answer": "Answer", "citation_numbers": []}'),
            "Generated question requires at least one citation number",
        ),
        (
            ('{"text": "Question", "expected_answer": "Answer", "citation_numbers": [true]}'),
            "Generated question citation_numbers must contain integers",
        ),
        (
            ('{"text": "Question", "expected_answer": "Answer", "citation_numbers": [0]}'),
            "Generated question citation numbers must be positive",
        ),
        (
            ('{"text": "Question", "expected_answer": "Answer", "citation_numbers": [1, 1]}'),
            "Generated question citation numbers must be unique",
        ),
    ],
)
def test_parse_question_generation_response_rejects_invalid_question(
    question_json: str,
    message: str,
) -> None:
    response = f'{{"questions": [{question_json}]}}'

    with pytest.raises(ValueError, match=message):
        parse_question_generation_response(response)


@pytest.mark.parametrize(
    ("field", "message"),
    [
        ("text", "Generated question text must not be blank"),
        (
            "expected_answer",
            "Generated question expected_answer must not be blank",
        ),
    ],
)
@pytest.mark.parametrize("value", ["", "   "])
def test_parse_question_generation_response_rejects_blank_text(
    field: str,
    message: str,
    value: str,
) -> None:
    question = {
        "text": "Question",
        "expected_answer": "Answer",
        "citation_numbers": [1],
    }
    question[field] = value

    response = json.dumps({"questions": [question]})

    with pytest.raises(ValueError, match=message):
        parse_question_generation_response(response)


def test_parse_question_generation_response_rejects_duplicate_questions() -> None:
    response = """
    {
      "questions": [
        {
          "text": "What is an embedding?",
          "expected_answer": "First answer.",
          "citation_numbers": [1]
        },
        {
          "text": "  WHAT IS AN EMBEDDING?  ",
          "expected_answer": "Second answer.",
          "citation_numbers": [2]
        }
      ]
    }
    """

    with pytest.raises(
        ValueError,
        match="Generated question texts must be unique",
    ):
        parse_question_generation_response(response)


def test_question_generation_result_preserves_drafts_and_prompts() -> None:
    draft = GeneratedQuestionDraft(
        number=1,
        text="What is an embedding?",
        expected_answer="A numeric representation of text.",
        citation_numbers=(1,),
    )
    prompt = PromptReference(
        name="question-generation.system-json",
        version=1,
        fingerprint="a" * 64,
    )

    result = QuestionGenerationResult(
        questions=(draft,),
        prompt_references=(prompt,),
    )

    assert result.questions == (draft,)
    assert result.prompt_references == (prompt,)


def test_question_generation_result_requires_questions() -> None:
    prompt = PromptReference(
        name="question-generation.system-json",
        version=1,
        fingerprint="a" * 64,
    )

    with pytest.raises(
        ValueError,
        match="Question generation result requires at least one question",
    ):
        QuestionGenerationResult(
            questions=(),
            prompt_references=(prompt,),
        )


def test_question_generation_result_requires_prompt_reference() -> None:
    draft = GeneratedQuestionDraft(
        number=1,
        text="What is an embedding?",
        expected_answer="A numeric representation of text.",
        citation_numbers=(1,),
    )

    with pytest.raises(
        ValueError,
        match="Question generation result requires at least one prompt reference",
    ):
        QuestionGenerationResult(
            questions=(draft,),
            prompt_references=(),
        )
