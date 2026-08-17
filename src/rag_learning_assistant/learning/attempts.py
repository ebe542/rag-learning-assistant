"""Immutable records of completed study-question attempts."""

import re
from dataclasses import dataclass
from datetime import datetime
from uuid import UUID

from rag_learning_assistant.generation import Citation
from rag_learning_assistant.learning.feedback import (
    AnswerEvaluation,
)
from rag_learning_assistant.learning.progress import (
    QuestionProgress,
    ReviewRating,
)


@dataclass(frozen=True, slots=True)
class StudyAttempt:
    """Preserve one learner answer and its resulting review schedule."""

    id: UUID
    document_id: UUID
    question_bank_identity_fingerprint: str
    question_number: int
    question_text: str
    answer_text: str
    expected_answer: str
    citations: tuple[Citation, ...]
    rating: ReviewRating
    answered_at: datetime
    resulting_progress: QuestionProgress
    evaluation: AnswerEvaluation | None = None

    def __post_init__(self) -> None:
        if not self.question_text.strip():
            raise ValueError("Attempt question text must not be blank")

        if not self.answer_text.strip():
            raise ValueError("Attempt answer text must not be blank")

        if not self.expected_answer.strip():
            raise ValueError("Attempt expected answer must not be blank")

        if (
            re.fullmatch(
                r"[0-9a-f]{64}",
                self.question_bank_identity_fingerprint,
            )
            is None
        ):
            raise ValueError(
                "Attempt question bank identity fingerprint must be a lowercase SHA-256 digest"
            )

        if self.question_number < 1:
            raise ValueError("Attempt question number must be positive")

        if not self.citations:
            raise ValueError("Study attempt requires at least one citation")

        citation_numbers = [citation.number for citation in self.citations]
        if len(set(citation_numbers)) != len(citation_numbers):
            raise ValueError("Study attempt citation numbers must be unique")

        if self.answered_at.tzinfo is None or self.answered_at.utcoffset() is None:
            raise ValueError("Attempt answer timestamp must include a timezone")

        if self.resulting_progress.document_id != self.document_id:
            raise ValueError("Attempt and progress document IDs must match")

        if (
            self.resulting_progress.question_bank_identity_fingerprint
            != self.question_bank_identity_fingerprint
        ):
            raise ValueError("Attempt and progress question bank identities must match")

        if self.resulting_progress.question_number != self.question_number:
            raise ValueError("Attempt and progress question numbers must match")

        if self.resulting_progress.last_reviewed_at != self.answered_at:
            raise ValueError("Attempt answer timestamp must match progress review timestamp")
