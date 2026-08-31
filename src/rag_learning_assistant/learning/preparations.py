"""Persisted requests awaiting learning-package preparation."""

from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from uuid import UUID

from rag_learning_assistant.learning.languages import LearningLanguage


class PackagePreparationStatus(StrEnum):
    """Describe work that has not entered the package checkpoints yet."""

    PENDING = "pending"
    INDEXING = "indexing"
    SUMMARIZING = "summarizing"
    GENERATING_QUESTIONS = "generating_questions"
    FAILED = "failed"


def _utc_now() -> datetime:
    return datetime.now(UTC)


@dataclass(frozen=True, slots=True)
class PackagePreparation:
    """Describe one safely stored PDF waiting for preparation."""

    id: UUID
    name: str
    source_filename: str
    stored_filename: str
    question_count: int
    size_bytes: int
    content_sha256: str | None = None
    learning_language: LearningLanguage = LearningLanguage.SAME_AS_DOCUMENT
    status: PackagePreparationStatus = PackagePreparationStatus.PENDING
    created_at: datetime = field(default_factory=_utc_now)
    updated_at: datetime | None = None
    lease_token: UUID | None = None
    lease_expires_at: datetime | None = None
    failure_message: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.learning_language, LearningLanguage):
            raise ValueError("Package preparation language must be a supported learning language")
        if not self.name.strip():
            raise ValueError("Package preparation name must not be blank")
        if not self.source_filename.strip():
            raise ValueError("Package preparation source filename must not be blank")
        if self.stored_filename != f"{self.id}.pdf":
            raise ValueError("Stored PDF filename must be derived from the preparation ID")
        if not 1 <= self.question_count <= 50:
            raise ValueError("Question count must be between 1 and 50")
        if self.size_bytes < 1:
            raise ValueError("Stored PDF must not be empty")
        if self.content_sha256 is not None and (
            len(self.content_sha256) != 64
            or any(character not in "0123456789abcdef" for character in self.content_sha256)
        ):
            raise ValueError("Package preparation hash must be a lowercase SHA-256 digest")
        updated_at = self.updated_at
        if updated_at is None:
            updated_at = self.created_at
            object.__setattr__(self, "updated_at", updated_at)
        if self.created_at.tzinfo is None or updated_at.tzinfo is None:
            raise ValueError("Package preparation timestamps must be timezone-aware")
        if (self.lease_token is None) != (self.lease_expires_at is None):
            raise ValueError("Package preparation lease fields must be set together")
        if self.lease_expires_at is not None and self.lease_expires_at.tzinfo is None:
            raise ValueError("Package preparation lease expiry must be timezone-aware")
        active_statuses = {
            PackagePreparationStatus.INDEXING,
            PackagePreparationStatus.SUMMARIZING,
            PackagePreparationStatus.GENERATING_QUESTIONS,
        }
        if self.status in active_statuses and self.lease_token is None:
            raise ValueError("Active package preparation requires a lease")
        if self.status not in active_statuses and self.lease_token is not None:
            raise ValueError("Inactive package preparation must not retain a lease")
        if self.status is PackagePreparationStatus.FAILED:
            if self.failure_message is None or not self.failure_message.strip():
                raise ValueError("Failed package preparation requires an error message")
        elif self.failure_message is not None:
            raise ValueError("Only failed package preparation may contain an error message")
