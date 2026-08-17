import math
from dataclasses import replace

import pytest

from rag_learning_assistant.generation import PromptReference
from rag_learning_assistant.learning import (
    AnswerEvaluation,
    AnswerVerdict,
)


def test_answer_evaluation_preserves_structured_feedback() -> None:
    evaluation = AnswerEvaluation(
        verdict=AnswerVerdict.PARTIALLY_CORRECT,
        score=0.7,
        feedback=(
            "The answer identifies retrieval, but does not explain "
            "that it happens before generation."
        ),
        missing_concepts=("Retrieval happens before generation.",),
    )

    assert evaluation.verdict is AnswerVerdict.PARTIALLY_CORRECT
    assert evaluation.score == 0.7
    assert evaluation.feedback.startswith("The answer identifies retrieval")
    assert evaluation.missing_concepts == ("Retrieval happens before generation.",)


def build_evaluation() -> AnswerEvaluation:
    return AnswerEvaluation(
        verdict=AnswerVerdict.PARTIALLY_CORRECT,
        score=0.7,
        feedback="The answer is incomplete.",
        missing_concepts=("Retrieval happens before generation.",),
    )


@pytest.mark.parametrize(
    "score",
    [
        -0.01,
        1.01,
        math.nan,
        math.inf,
        -math.inf,
    ],
)
def test_answer_evaluation_requires_finite_normalized_score(
    score: float,
) -> None:
    with pytest.raises(
        ValueError,
        match="Evaluation score must be between 0 and 1",
    ):
        replace(build_evaluation(), score=score)


@pytest.mark.parametrize("feedback", ["", "   "])
def test_answer_evaluation_rejects_blank_feedback(
    feedback: str,
) -> None:
    with pytest.raises(
        ValueError,
        match="Evaluation feedback must not be blank",
    ):
        replace(build_evaluation(), feedback=feedback)


def test_answer_evaluation_rejects_blank_missing_concept() -> None:
    with pytest.raises(
        ValueError,
        match="Missing concepts must not be blank",
    ):
        replace(
            build_evaluation(),
            missing_concepts=("   ",),
        )


def test_answer_evaluation_rejects_duplicate_missing_concepts() -> None:
    with pytest.raises(
        ValueError,
        match="Missing concepts must be unique",
    ):
        replace(
            build_evaluation(),
            missing_concepts=(
                "Source grounding",
                " source grounding ",
            ),
        )


def test_correct_evaluation_rejects_missing_concepts() -> None:
    with pytest.raises(
        ValueError,
        match="Correct evaluation must not contain missing concepts",
    ):
        replace(
            build_evaluation(),
            verdict=AnswerVerdict.CORRECT,
        )


def test_answer_evaluation_rejects_duplicate_prompt_references() -> None:
    reference = PromptReference(
        name="answer-evaluation.system-json",
        version=1,
        fingerprint="d" * 64,
    )

    with pytest.raises(
        ValueError,
        match="Evaluation prompt references must be unique",
    ):
        replace(
            build_evaluation(),
            prompt_references=(
                reference,
                reference,
            ),
        )
