"""Application models and coordination for learning progress reports."""

from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from typing import Protocol
from uuid import UUID

from rag_learning_assistant.application.package_study import (
    LearningPackageNotFoundError,
    LearningPackageNotReadyError,
)
from rag_learning_assistant.learning import (
    AnswerVerdict,
    LearningPackage,
    LearningPackageStatus,
    QuestionBank,
    QuestionProgress,
    StudyAttempt,
)


@dataclass(frozen=True, slots=True)
class LearningProgressReport:
    """Summarize study activity for one user-facing learning package."""

    package_name: str
    total_question_count: int
    answered_question_count: int
    due_question_count: int
    attempt_count: int
    incorrect_attempt_count: int
    partially_correct_attempt_count: int
    correct_attempt_count: int
    difficult_concepts: tuple[tuple[str, int], ...]
    last_studied_at: datetime | None
    next_due_at: datetime | None
    unclassified_attempt_count: int

    def __post_init__(self) -> None:
        """Reject invalid report data at the application boundary."""

        if not self.package_name.strip():
            raise ValueError("Progress package name must not be blank")
        counts = (
            self.total_question_count,
            self.answered_question_count,
            self.due_question_count,
            self.attempt_count,
            self.incorrect_attempt_count,
            self.partially_correct_attempt_count,
            self.correct_attempt_count,
            self.unclassified_attempt_count,
        )

        # Report values are aggregated from several persistence sources. Rejecting
        # invalid counts here keeps every CLI or future UI consumer consistent.
        if any(count < 0 for count in counts):
            raise ValueError("Progress counts must not be negative")

        classified_attempt_count = (
            self.incorrect_attempt_count
            + self.partially_correct_attempt_count
            + self.correct_attempt_count
            + self.unclassified_attempt_count
        )

        if classified_attempt_count != self.attempt_count:
            raise ValueError("Progress attempt counts must equal total attempts")

    @property
    def answered_rate(self) -> float:
        """Return the share of distinct questions answered at least once."""

        if self.total_question_count == 0:
            return 0.0

        return self.answered_question_count / self.total_question_count

    @property
    def correct_attempt_rate(self) -> float:
        """Return the share of attempts evaluated as correct."""

        if self.attempt_count == 0:
            return 0.0

        return self.correct_attempt_count / self.attempt_count


class LearningProgressPackageLookup(Protocol):
    """Resolve the user-facing package selected for a report."""

    def find_by_name(
        self,
        name: str,
    ) -> LearningPackage | None: ...


class LearningProgressQuestionBankLookup(Protocol):
    """Load the exact active question bank of a package."""

    def get_document_bank(
        self,
        document_id: UUID,
        identity_fingerprint: str,
    ) -> QuestionBank: ...


class LearningProgressReader(Protocol):
    """Load the current schedule of one study question."""

    def find(
        self,
        document_id: UUID,
        question_bank_identity_fingerprint: str,
        question_number: int,
    ) -> QuestionProgress | None: ...


class LearningAttemptReader(Protocol):
    """Load the immutable attempt history of one question."""

    def list_question(
        self,
        document_id: UUID,
        question_bank_identity_fingerprint: str,
        question_number: int,
    ) -> list[StudyAttempt]: ...


class LearningProgressService:
    """Aggregate study progress for one user-facing package."""

    def __init__(
        self,
        packages: LearningProgressPackageLookup,
        banks: LearningProgressQuestionBankLookup,
        progress: LearningProgressReader,
        attempts: LearningAttemptReader,
    ) -> None:
        self.packages = packages
        self.banks = banks
        self.progress = progress
        self.attempts = attempts

    def report(
        self,
        package_name: str,
        *,
        as_of: datetime,
    ) -> LearningProgressReport:
        """Build a current progress report for one package."""

        if not package_name.strip():
            raise ValueError("Progress package name must not be blank")

        if as_of.tzinfo is None or as_of.utcoffset() is None:
            raise ValueError("Progress timestamp must include a timezone")

        package = self.packages.find_by_name(package_name)

        if package is None:
            raise LearningPackageNotFoundError(f"Learning package does not exist: {package_name}")

        if (
            package.status is not LearningPackageStatus.READY
            or package.question_bank_identity_fingerprint is None
        ):
            raise LearningPackageNotReadyError(f"Learning package is not ready: {package.name}")

        fingerprint = package.question_bank_identity_fingerprint
        bank = self.banks.get_document_bank(
            package.document_id,
            fingerprint,
        )

        question_attempts = [
            self.attempts.list_question(
                package.document_id,
                fingerprint,
                question.number,
            )
            for question in bank.questions
        ]
        question_progress = [
            self.progress.find(
                package.document_id,
                fingerprint,
                question.number,
            )
            for question in bank.questions
        ]
        attempts = [attempt for history in question_attempts for attempt in history]

        incorrect_count = sum(
            attempt.evaluation is not None and attempt.evaluation.verdict is AnswerVerdict.INCORRECT
            for attempt in attempts
        )
        partially_correct_count = sum(
            attempt.evaluation is not None
            and attempt.evaluation.verdict is AnswerVerdict.PARTIALLY_CORRECT
            for attempt in attempts
        )
        correct_count = sum(
            attempt.evaluation is not None and attempt.evaluation.verdict is AnswerVerdict.CORRECT
            for attempt in attempts
        )
        unclassified_count = sum(attempt.evaluation is None for attempt in attempts)

        concept_counts = Counter(
            concept
            for attempt in attempts
            if attempt.evaluation is not None
            for concept in attempt.evaluation.missing_concepts
        )

        # A new question has no schedule yet and is therefore due immediately.
        due_question_count = sum(
            progress is None or progress.due_at <= as_of for progress in question_progress
        )
        next_due_at = min(
            (as_of if progress is None else progress.due_at for progress in question_progress),
            default=None,
        )

        return LearningProgressReport(
            package_name=package.name,
            total_question_count=len(bank.questions),
            answered_question_count=sum(bool(history) for history in question_attempts),
            due_question_count=due_question_count,
            attempt_count=len(attempts),
            incorrect_attempt_count=incorrect_count,
            partially_correct_attempt_count=(partially_correct_count),
            correct_attempt_count=correct_count,
            difficult_concepts=tuple(
                sorted(
                    concept_counts.items(),
                    key=lambda item: (
                        -item[1],
                        item[0].casefold(),
                    ),
                )
            ),
            last_studied_at=max(
                (attempt.answered_at for attempt in attempts),
                default=None,
            ),
            next_due_at=next_due_at,
            unclassified_attempt_count=unclassified_count,
        )
