import pytest

from rag_learning_assistant.generation.response_parser import (
    parse_generation_response,
)


def test_parse_generation_response_returns_typed_result() -> None:
    raw_response = """
    {
      "text": "A function groups reusable instructions.",
      "citation_numbers": [1, 3]
    }
    """

    result = parse_generation_response(raw_response)

    assert result.text == "A function groups reusable instructions."
    assert result.citation_numbers == (1, 3)


@pytest.mark.parametrize(
    ("raw_response", "message"),
    [
        ("not JSON", "Model response must be valid JSON"),
        ("[]", "Model response must be a JSON object"),
        (
            '{"text": "Answer"}',
            "Model response must contain exactly",
        ),
        (
            """
            {
              "text": "Answer",
              "citation_numbers": [],
              "unexpected": true
            }
            """,
            "Model response must contain exactly",
        ),
        (
            '{"text": 42, "citation_numbers": []}',
            "Model response text must be a string",
        ),
        (
            '{"text": "Answer", "citation_numbers": "1"}',
            "Model citation_numbers must be an array",
        ),
        (
            '{"text": "Answer", "citation_numbers": [true]}',
            "Model citation_numbers must contain integers",
        ),
        (
            '{"text": "Answer", "citation_numbers": [1, "2"]}',
            "Model citation_numbers must contain integers",
        ),
    ],
)
def test_parse_generation_response_rejects_invalid_schema(
    raw_response: str,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        parse_generation_response(raw_response)
