"""Learning progress and review scheduling domain models."""

import re
from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from uuid import UUID


class ReviewRating(StrEnum):
    """Describe how confidently a learner answered a study question."""

    AGAIN = "again"
    HARD = "hard"
    GOOD = "good"
    EASY = "easy"


@dataclass(frozen=True, slots=True)
class QuestionProgress:
    """Track the review schedule of one versioned study question."""

    document_id: UUID
    question_bank_identity_fingerprint: str
    question_number: int
    repetition_count: int
    interval_days: int
    ease_factor: float
    due_at: datetime
    last_reviewed_at: datetime | None

    def __post_init__(self) -> None:
        if (
            re.fullmatch(
                r"[0-9a-f]{64}",
                self.question_bank_identity_fingerprint,
            )
            is None
        ):
            raise ValueError(
                "Question bank identity fingerprint must be a lowercase SHA-256 digest"
            )

        if self.question_number <= 0:
            raise ValueError("Question number must be positive")

        if self.repetition_count < 0:
            raise ValueError("Repetition count must not be negative")

        if self.interval_days < 0:
            raise ValueError("Review interval must not be negative")

        if self.ease_factor < 1.3:
            raise ValueError("Ease factor must be at least 1.3")

        if self.due_at.tzinfo is None or self.due_at.utcoffset() is None:
            raise ValueError("Due date must include a timezone")

        if self.last_reviewed_at is not None:
            if self.last_reviewed_at.tzinfo is None or self.last_reviewed_at.utcoffset() is None:
                raise ValueError("Last review date must include a timezone")

            if self.due_at < self.last_reviewed_at:
                raise ValueError("Due date must not be earlier than the last review date")
