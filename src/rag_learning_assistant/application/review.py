"""Application logic for scheduling study-question reviews."""

from dataclasses import dataclass, replace
from datetime import datetime, timedelta
from typing import Protocol
from uuid import UUID

from rag_learning_assistant.learning import (
    QuestionBank,
    QuestionProgress,
    ReviewRating,
    StudyQuestion,
)


class ReviewQuestionBankLookup(Protocol):
    """Load one exact persisted question bank."""

    def get_document_bank(
        self,
        document_id: UUID,
        identity_fingerprint: str,
    ) -> QuestionBank: ...


class ReviewProgressRepository(Protocol):
    """Load and persist the current state of one study question."""

    def find(
        self,
        document_id: UUID,
        question_bank_identity_fingerprint: str,
        question_number: int,
    ) -> QuestionProgress | None: ...

    def save(self, progress: QuestionProgress) -> None: ...


class StudyQuestionNotFoundError(LookupError):
    """Raised when a question number is absent from a selected bank."""


@dataclass(frozen=True, slots=True)
class DueQuestion:
    """Pair a due study question with its optional review progress."""

    question: StudyQuestion
    progress: QuestionProgress | None


class ReviewScheduler:
    """Calculate the next immutable review state."""

    def review(
        self,
        progress: QuestionProgress,
        rating: ReviewRating,
        *,
        reviewed_at: datetime,
    ) -> QuestionProgress:
        """Apply one learner rating to a question's schedule."""

        if reviewed_at.tzinfo is None or reviewed_at.utcoffset() is None:
            raise ValueError("Review timestamp must include a timezone")

        if progress.last_reviewed_at is not None and reviewed_at < progress.last_reviewed_at:
            raise ValueError("Review timestamp must not precede the previous review")

        if rating is ReviewRating.AGAIN:
            repetition_count = 0
            interval_days = 0
            ease_factor = max(1.3, progress.ease_factor - 0.2)
            due_at = reviewed_at + timedelta(minutes=10)
        elif rating is ReviewRating.HARD:
            repetition_count = progress.repetition_count + 1
            interval_days = (
                1 if progress.interval_days == 0 else max(1, round(progress.interval_days * 1.2))
            )
            ease_factor = max(1.3, progress.ease_factor - 0.15)
            due_at = reviewed_at + timedelta(days=interval_days)
        elif rating is ReviewRating.GOOD:
            repetition_count = progress.repetition_count + 1

            if progress.repetition_count == 0:
                interval_days = 1
            elif progress.repetition_count == 1:
                interval_days = 6
            else:
                interval_days = max(
                    1,
                    round(progress.interval_days * progress.ease_factor),
                )

            ease_factor = progress.ease_factor
            due_at = reviewed_at + timedelta(days=interval_days)
        else:
            repetition_count = progress.repetition_count + 1
            interval_days = (
                4
                if progress.interval_days == 0
                else max(
                    4,
                    round(progress.interval_days * progress.ease_factor * 1.3),
                )
            )
            ease_factor = progress.ease_factor + 0.15
            due_at = reviewed_at + timedelta(days=interval_days)

        return replace(
            progress,
            repetition_count=repetition_count,
            interval_days=interval_days,
            ease_factor=ease_factor,
            due_at=due_at,
            last_reviewed_at=reviewed_at,
        )


class ReviewService:
    """Coordinate question lookup, scheduling, and progress persistence."""

    def __init__(
        self,
        banks: ReviewQuestionBankLookup,
        progress: ReviewProgressRepository,
        scheduler: ReviewScheduler,
    ) -> None:
        self.banks = banks
        self.progress = progress
        self.scheduler = scheduler

    def list_due(
        self,
        document_id: UUID,
        question_bank_identity_fingerprint: str,
        *,
        as_of: datetime,
        limit: int,
    ) -> list[DueQuestion]:
        """Return due questions from one exact persisted bank."""

        if limit < 1:
            raise ValueError("limit must be positive")

        if as_of.tzinfo is None or as_of.utcoffset() is None:
            raise ValueError("Due-query timestamp must include a timezone")

        bank = self.banks.get_document_bank(
            document_id,
            question_bank_identity_fingerprint,
        )
        due: list[DueQuestion] = []

        for question in bank.questions:
            current = self.progress.find(
                document_id,
                question_bank_identity_fingerprint,
                question.number,
            )

            if current is None or current.due_at <= as_of:
                due.append(
                    DueQuestion(
                        question=question,
                        progress=current,
                    )
                )

        def priority(item: DueQuestion) -> tuple[int, datetime, int]:
            if item.progress is None:
                # New questions follow scheduled reviews and retain bank order.
                return (1, as_of, item.question.number)

            # The oldest due date receives the highest review priority.
            return (
                0,
                item.progress.due_at,
                item.question.number,
            )

        due.sort(key=priority)
        return due[:limit]

    def prepare_review(
        self,
        document_id: UUID,
        question_bank_identity_fingerprint: str,
        question_number: int,
        rating: ReviewRating,
        *,
        reviewed_at: datetime,
    ) -> QuestionProgress:
        """Calculate a review result without persisting it."""

        bank = self.banks.get_document_bank(
            document_id,
            question_bank_identity_fingerprint,
        )
        question = next(
            (candidate for candidate in bank.questions if candidate.number == question_number),
            None,
        )

        if question is None:
            raise StudyQuestionNotFoundError(
                "Study question does not exist: "
                f"{document_id}/"
                f"{question_bank_identity_fingerprint}/"
                f"{question_number}"
            )

        current = self.progress.find(
            document_id,
            question_bank_identity_fingerprint,
            question_number,
        )

        if current is None:
            # Missing progress means the persisted question has never been
            # reviewed and is therefore due immediately.
            current = QuestionProgress(
                document_id=document_id,
                question_bank_identity_fingerprint=(question_bank_identity_fingerprint),
                question_number=question_number,
                repetition_count=0,
                interval_days=0,
                ease_factor=2.5,
                due_at=reviewed_at,
                last_reviewed_at=None,
            )

        return self.scheduler.review(
            current,
            rating,
            reviewed_at=reviewed_at,
        )

    def record_review(
        self,
        document_id: UUID,
        question_bank_identity_fingerprint: str,
        question_number: int,
        rating: ReviewRating,
        *,
        reviewed_at: datetime,
    ) -> QuestionProgress:
        """Calculate and persist one question's updated schedule."""

        reviewed = self.prepare_review(
            document_id,
            question_bank_identity_fingerprint,
            question_number,
            rating,
            reviewed_at=reviewed_at,
        )
        self.progress.save(reviewed)
        return reviewed
