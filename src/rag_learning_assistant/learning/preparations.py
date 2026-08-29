"""Persisted requests awaiting learning-package preparation."""

from dataclasses import dataclass
from enum import StrEnum
from uuid import UUID


class PackagePreparationStatus(StrEnum):
    """Describe work that has not entered the package checkpoints yet."""

    PENDING = "pending"


@dataclass(frozen=True, slots=True)
class PackagePreparation:
    """Describe one safely stored PDF waiting for preparation."""

    id: UUID
    name: str
    source_filename: str
    stored_filename: str
    question_count: int
    size_bytes: int
    status: PackagePreparationStatus = PackagePreparationStatus.PENDING

    def __post_init__(self) -> None:
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
