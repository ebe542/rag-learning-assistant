"""Application logic for recording completed study-question attempts."""

from collections.abc import Callable
from dataclasses import replace
from datetime import datetime
from typing import Protocol
from uuid import UUID

from rag_learning_assistant.application.answer_evaluation import (
    EvaluatedStudyAnswer,
)
from rag_learning_assistant.application.review import (
    DueQuestion,
    StudyQuestionNotFoundError,
)
from rag_learning_assistant.learning import (
    QuestionBank,
    QuestionProgress,
    ReviewRating,
    StudyAttempt,
    StudyQuestion,
)


class StudySessionQuestionBankLookup(Protocol):
    """Load the exact question bank used by a study session."""

    def get_document_bank(
        self,
        document_id: UUID,
        identity_fingerprint: str,
    ) -> QuestionBank: ...


class StudySessionReviewer(Protocol):
    """Update the review schedule for one answered question."""

    def list_due(
        self,
        document_id: UUID,
        question_bank_identity_fingerprint: str,
        *,
        as_of: datetime,
        limit: int,
    ) -> list[DueQuestion]: ...

    def prepare_review(
        self,
        document_id: UUID,
        question_bank_identity_fingerprint: str,
        question_number: int,
        rating: ReviewRating,
        *,
        reviewed_at: datetime,
    ) -> QuestionProgress: ...


class StudyAnswerEvaluator(Protocol):
    """Evaluate one written answer before scheduling its review."""

    def evaluate(
        self,
        question: StudyQuestion,
        answer_text: str,
    ) -> EvaluatedStudyAnswer: ...


class StudySessionRecorder(Protocol):
    """Atomically persist one attempt and its resulting progress."""

    def record(self, attempt: StudyAttempt) -> None: ...


class StudySessionService:
    """Coordinate answering, scheduling, and attempt persistence."""

    def __init__(
        self,
        banks: StudySessionQuestionBankLookup,
        reviewer: StudySessionReviewer,
        attempts: StudySessionRecorder,
        attempt_id_factory: Callable[[], UUID],
        evaluator: StudyAnswerEvaluator | None = None,
    ) -> None:
        self.banks = banks
        self.reviewer = reviewer
        self.attempts = attempts
        self.attempt_id_factory = attempt_id_factory
        self.evaluator = evaluator

    def next_due(
        self,
        document_id: UUID,
        question_bank_identity_fingerprint: str,
        *,
        as_of: datetime,
    ) -> DueQuestion | None:
        """Return the highest-priority due question, if one exists."""

        due = self.reviewer.list_due(
            document_id,
            question_bank_identity_fingerprint,
            as_of=as_of,
            limit=1,
        )
        return due[0] if due else None

    def record_answer(
        self,
        document_id: UUID,
        question_bank_identity_fingerprint: str,
        question_number: int,
        *,
        answer_text: str,
        answered_at: datetime,
        rating: ReviewRating | None = None,
    ) -> StudyAttempt:
        """Record one answer and return its resulting immutable history entry."""

        if not answer_text.strip():
            raise ValueError("Study answer must not be blank")

        if answered_at.tzinfo is None or answered_at.utcoffset() is None:
            raise ValueError("Study answer timestamp must include a timezone")

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

        evaluation = None

        if self.evaluator is not None:
            evaluated = self.evaluator.evaluate(
                question,
                answer_text,
            )
            effective_rating = evaluated.rating

            # Persist the complete prompt chain, including the application
            # prompt and every generator or repair prompt actually used.
            evaluation = replace(
                evaluated.evaluation,
                prompt_references=(evaluated.prompt_references),
            )
        elif rating is not None:
            effective_rating = rating
        else:
            raise RuntimeError("Automatic answer evaluation is not configured")

        progress = self.reviewer.prepare_review(
            document_id,
            question_bank_identity_fingerprint,
            question_number,
            effective_rating,
            reviewed_at=answered_at,
        )

        attempt = StudyAttempt(
            id=self.attempt_id_factory(),
            document_id=document_id,
            question_bank_identity_fingerprint=(question_bank_identity_fingerprint),
            question_number=question_number,
            question_text=question.text,
            answer_text=answer_text,
            expected_answer=question.expected_answer,
            citations=question.citations,
            rating=effective_rating,
            answered_at=answered_at,
            resulting_progress=progress,
            evaluation=evaluation,
        )
        self.attempts.record(attempt)
        return attempt
