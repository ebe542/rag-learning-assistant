"""Parse structured written-answer evaluations from model output."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from rag_learning_assistant.learning.feedback import (
        AnswerEvaluation,
    )


def parse_answer_evaluation(
    response: str,
) -> AnswerEvaluation:
    """Convert one structured model response into a domain result."""

    # Importing learning while generation is initialized would create a cycle:
    # learning attempts already depend on generation's Citation model.
    from rag_learning_assistant.learning.feedback import (
        AnswerEvaluation,
        AnswerVerdict,
    )

    try:
        payload = json.loads(response)
    except json.JSONDecodeError as exc:
        raise ValueError("Model evaluation must be valid JSON") from exc

    if not isinstance(payload, dict):
        raise ValueError("Model evaluation must be a JSON object")

    expected_fields = {
        "verdict",
        "score",
        "feedback",
        "missing_concepts",
    }
    if set(payload) != expected_fields:
        raise ValueError(
            "Model evaluation must contain exactly: verdict, score, feedback, missing_concepts"
        )

    verdict_value = payload["verdict"]
    if not isinstance(verdict_value, str):
        raise ValueError("Model evaluation verdict is invalid")

    try:
        verdict = AnswerVerdict(verdict_value)
    except ValueError as exc:
        raise ValueError("Model evaluation verdict is invalid") from exc

    score = payload["score"]
    if isinstance(score, bool) or not isinstance(score, (int, float)):
        raise ValueError("Model evaluation score must be a number")

    feedback = payload["feedback"]
    if not isinstance(feedback, str):
        raise ValueError("Model evaluation feedback must be a string")

    missing_concepts = payload["missing_concepts"]
    if not isinstance(missing_concepts, list):
        raise ValueError("Model evaluation missing_concepts must be an array")

    if any(not isinstance(concept, str) for concept in missing_concepts):
        raise ValueError("Model evaluation missing_concepts must contain strings")

    return AnswerEvaluation(
        verdict=verdict,
        score=float(score),
        feedback=feedback,
        missing_concepts=tuple(missing_concepts),
    )
