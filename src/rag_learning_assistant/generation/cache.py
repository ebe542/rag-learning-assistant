"""Cache contracts for resumable document generation."""

from dataclasses import dataclass
from typing import Protocol

from rag_learning_assistant.generation.models import (
    GenerationResult,
)


@dataclass(frozen=True, slots=True)
class CachedSummaryBatch:
    """One successfully generated map result and its source range."""

    identity_fingerprint: str
    batch_number: int
    first_context_number: int
    last_context_number: int
    result: GenerationResult

    def __post_init__(self) -> None:
        is_generation_fingerprint = len(self.identity_fingerprint) == 64 and all(
            character in "0123456789abcdef" for character in self.identity_fingerprint
        )
        if not is_generation_fingerprint:
            raise ValueError("Generation fingerprint must be a lowercase SHA-256 hex digest")

        if self.batch_number < 1:
            raise ValueError("Batch number must be positive")

        if self.first_context_number < 1 or self.last_context_number < 1:
            raise ValueError("Context numbers must be positive")

        if self.last_context_number < self.first_context_number:
            raise ValueError("Last context number must not precede first")

        if any(
            citation_number < self.first_context_number
            or citation_number > self.last_context_number
            for citation_number in self.result.citation_numbers
        ):
            raise ValueError("Cached citation does not belong to its batch")


class SummaryBatchCache(Protocol):
    """Load and persist successful map-generation results."""

    def find_batch(
        self,
        identity_fingerprint: str,
        batch_number: int,
    ) -> CachedSummaryBatch | None:
        """Return one cached batch or None."""

        ...

    def save_batch(
        self,
        batch: CachedSummaryBatch,
    ) -> None:
        """Persist one completed batch idempotently."""

        ...
