import pytest

from rag_learning_assistant.generation import (
    parse_answer_evaluation,
)
from rag_learning_assistant.learning import (
    AnswerEvaluation,
    AnswerVerdict,
)


def test_parse_answer_evaluation_returns_validated_domain_model() -> None:
    response = """
    {
      "verdict": "partially_correct",
      "score": 0.7,
      "feedback": "The answer identifies retrieval but misses its order.",
      "missing_concepts": [
        "Retrieval happens before generation."
      ]
    }
    """

    evaluation = parse_answer_evaluation(response)

    assert evaluation == AnswerEvaluation(
        verdict=AnswerVerdict.PARTIALLY_CORRECT,
        score=0.7,
        feedback=("The answer identifies retrieval but misses its order."),
        missing_concepts=("Retrieval happens before generation.",),
    )


@pytest.mark.parametrize(
    ("response", "message"),
    [
        (
            "not JSON",
            "Model evaluation must be valid JSON",
        ),
        (
            "[]",
            "Model evaluation must be a JSON object",
        ),
        (
            '{"verdict": "correct"}',
            "Model evaluation must contain exactly",
        ),
        (
            """
            {
              "verdict": "correct",
              "score": 1.0,
              "feedback": "Correct.",
              "missing_concepts": [],
              "rating": "easy"
            }
            """,
            "Model evaluation must contain exactly",
        ),
        (
            """
            {
              "verdict": "mostly_correct",
              "score": 0.8,
              "feedback": "Almost correct.",
              "missing_concepts": []
            }
            """,
            "Model evaluation verdict is invalid",
        ),
        (
            """
            {
              "verdict": "correct",
              "score": "1.0",
              "feedback": "Correct.",
              "missing_concepts": []
            }
            """,
            "Model evaluation score must be a number",
        ),
        (
            """
            {
              "verdict": "correct",
              "score": true,
              "feedback": "Correct.",
              "missing_concepts": []
            }
            """,
            "Model evaluation score must be a number",
        ),
        (
            """
            {
              "verdict": "correct",
              "score": 1.0,
              "feedback": 42,
              "missing_concepts": []
            }
            """,
            "Model evaluation feedback must be a string",
        ),
        (
            """
            {
              "verdict": "correct",
              "score": 1.0,
              "feedback": "Correct.",
              "missing_concepts": "none"
            }
            """,
            "Model evaluation missing_concepts must be an array",
        ),
        (
            """
            {
              "verdict": "partially_correct",
              "score": 0.7,
              "feedback": "Incomplete.",
              "missing_concepts": [42]
            }
            """,
            "Model evaluation missing_concepts must contain strings",
        ),
    ],
)
def test_parse_answer_evaluation_rejects_invalid_schema(
    response: str,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        parse_answer_evaluation(response)
