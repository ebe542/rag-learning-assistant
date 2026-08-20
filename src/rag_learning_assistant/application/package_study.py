"""Resolve user-facing learning packages for study sessions."""

from datetime import datetime
from typing import Protocol
from uuid import UUID

from rag_learning_assistant.application.review import DueQuestion
from rag_learning_assistant.learning import (
    LearningPackage,
    LearningPackageStatus,
    ReviewRating,
    StudyAttempt,
)


class LearningPackageLookup(Protocol):
    """Find a learning package by its user-facing name."""

    def find_by_name(
        self,
        name: str,
    ) -> LearningPackage | None: ...


class PackageStudySession(Protocol):
    """Select due questions through exact technical identities."""

    def next_due(
        self,
        document_id: UUID,
        question_bank_identity_fingerprint: str,
        *,
        as_of: datetime,
    ) -> DueQuestion | None: ...

    def record_answer(
        self,
        document_id: UUID,
        question_bank_identity_fingerprint: str,
        question_number: int,
        *,
        answer_text: str,
        answered_at: datetime,
        rating: ReviewRating | None = None,
    ) -> StudyAttempt: ...


class LearningPackageNotFoundError(LookupError):
    """Raised when a requested learning package does not exist."""


class LearningPackageNotReadyError(ValueError):
    """Raised when a learning package has no usable question bank."""


class LearningPackageStudyService:
    """Translate a package name into an existing study session."""

    def __init__(
        self,
        packages: LearningPackageLookup,
        sessions: PackageStudySession,
    ) -> None:
        self.packages = packages
        self.sessions = sessions

    def next_due(
        self,
        package_name: str,
        *,
        as_of: datetime,
    ) -> DueQuestion | None:
        """Return the next due question for one ready package."""

        package = self._get_ready_package(package_name)
        fingerprint = package.question_bank_identity_fingerprint

        # A ready package is guaranteed to reference a generated question bank.
        assert fingerprint is not None

        return self.sessions.next_due(
            package.document_id,
            fingerprint,
            as_of=as_of,
        )

    def record_answer(
        self,
        package_name: str,
        question_number: int,
        *,
        answer_text: str,
        answered_at: datetime,
        rating: ReviewRating | None = None,
    ) -> StudyAttempt:
        """Record a written answer for one ready learning package."""

        package = self._get_ready_package(package_name)
        fingerprint = package.question_bank_identity_fingerprint

        # A ready package is guaranteed to reference a generated question bank.
        assert fingerprint is not None

        return self.sessions.record_answer(
            package.document_id,
            fingerprint,
            question_number,
            answer_text=answer_text,
            answered_at=answered_at,
            rating=rating,
        )

    def _get_ready_package(
        self,
        package_name: str,
    ) -> LearningPackage:
        """Resolve a ready package before accessing its study data."""

        if not package_name.strip():
            raise ValueError("Learning package name must not be blank")

        package = self.packages.find_by_name(package_name)

        if package is None:
            raise LearningPackageNotFoundError(f"Learning package does not exist: {package_name}")

        if (
            package.status is not LearningPackageStatus.READY
            or package.question_bank_identity_fingerprint is None
        ):
            raise LearningPackageNotReadyError(f"Learning package is not ready: {package.name}")

        return package
