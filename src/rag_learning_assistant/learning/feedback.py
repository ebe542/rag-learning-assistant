"""Domain models for grounded written-answer evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from rag_learning_assistant.generation.prompts import (
        PromptReference,
    )


class AnswerVerdict(StrEnum):
    """Classify the factual quality of one written learner answer."""

    INCORRECT = "incorrect"
    PARTIALLY_CORRECT = "partially_correct"
    CORRECT = "correct"


@dataclass(frozen=True, slots=True)
class AnswerEvaluation:
    """Preserve validated feedback for one written answer."""

    verdict: AnswerVerdict
    score: float
    feedback: str
    missing_concepts: tuple[str, ...]
    prompt_references: tuple[PromptReference, ...] = ()

    def __post_init__(self) -> None:
        if not 0.0 <= self.score <= 1.0:
            raise ValueError("Evaluation score must be between 0 and 1")

        if not self.feedback.strip():
            raise ValueError("Evaluation feedback must not be blank")

        if any(not concept.strip() for concept in self.missing_concepts):
            raise ValueError("Missing concepts must not be blank")

        if self.verdict is AnswerVerdict.CORRECT and self.missing_concepts:
            raise ValueError("Correct evaluation must not contain missing concepts")

        if self.verdict is not AnswerVerdict.CORRECT and not self.missing_concepts:
            raise ValueError("Non-correct evaluation requires a missing concept")

        if len(set(self.prompt_references)) != len(self.prompt_references):
            raise ValueError("Evaluation prompt references must be unique")

        normalized_concepts = tuple(concept.strip().casefold() for concept in self.missing_concepts)
        if len(set(normalized_concepts)) != len(normalized_concepts):
            raise ValueError("Missing concepts must be unique")
