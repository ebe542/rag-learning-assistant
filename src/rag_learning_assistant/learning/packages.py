"""User-facing grouping of versioned learning material."""

from dataclasses import dataclass
from enum import StrEnum
from uuid import UUID

from rag_learning_assistant.learning.languages import LearningLanguage


def _is_lowercase_sha256(value: str) -> bool:
    """Recognize the canonical fingerprint format used by persisted identities."""

    return len(value) == 64 and all(character in "0123456789abcdef" for character in value)


class LearningPackageStatus(StrEnum):
    """Last successfully completed preparation step."""

    INDEXED = "indexed"
    SUMMARIZED = "summarized"
    READY = "ready"


@dataclass(frozen=True, slots=True)
class LearningPackage:
    """Connect one document with its active derived learning material."""

    id: UUID
    name: str
    document_id: UUID
    status: LearningPackageStatus
    summary_identity_fingerprint: str | None = None
    question_bank_identity_fingerprint: str | None = None
    learning_language: LearningLanguage = LearningLanguage.SAME_AS_DOCUMENT

    def __post_init__(self) -> None:
        if not isinstance(self.learning_language, LearningLanguage):
            raise ValueError("Learning package language must be a supported learning language")
        if not self.name.strip():
            raise ValueError("Learning package name must not be blank")

        if self.summary_identity_fingerprint is not None and not _is_lowercase_sha256(
            self.summary_identity_fingerprint
        ):
            raise ValueError("Summary identity must be a lowercase SHA-256 fingerprint")

        if self.question_bank_identity_fingerprint is not None and not _is_lowercase_sha256(
            self.question_bank_identity_fingerprint
        ):
            raise ValueError("Question-bank identity must be a lowercase SHA-256 fingerprint")

        if self.status is LearningPackageStatus.INDEXED:
            if (
                self.summary_identity_fingerprint is not None
                or self.question_bank_identity_fingerprint is not None
            ):
                raise ValueError("Indexed learning package must not reference derived material")
            return

        if self.summary_identity_fingerprint is None:
            raise ValueError("Summarized learning package requires a summary identity")

        if self.status is LearningPackageStatus.SUMMARIZED:
            if self.question_bank_identity_fingerprint is not None:
                raise ValueError("Summarized learning package must not reference a question bank")
            return

        if self.question_bank_identity_fingerprint is None:
            raise ValueError("Ready learning package requires a question-bank identity")
