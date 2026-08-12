"""Stable identities for reproducible generation runs."""

import json
from dataclasses import dataclass
from hashlib import sha256

from rag_learning_assistant.generation.prompts import (
    PromptReference,
)


@dataclass(frozen=True, slots=True)
class GenerationIdentity:
    """Identify one exact document-generation configuration."""

    model_name: str
    model_revision: str
    prompt_references: tuple[PromptReference, ...]
    max_map_new_tokens: int
    max_reduce_new_tokens: int
    max_batch_chars: int
    document_content_sha256: str

    def __post_init__(self) -> None:
        if not self.model_name.strip() or not self.model_revision.strip():
            raise ValueError("Generation model fields must not be blank")

        if not self.prompt_references:
            raise ValueError("Generation identity requires at least one prompt")

        if (
            self.max_map_new_tokens < 1
            or self.max_reduce_new_tokens < 1
            or self.max_batch_chars < 1
        ):
            raise ValueError("Generation limits must be positive")

        is_document_sha256 = len(self.document_content_sha256) == 64 and all(
            character in "0123456789abcdef" for character in self.document_content_sha256
        )
        if not is_document_sha256:
            raise ValueError("Document content hash must be a lowercase SHA-256 hex digest")

        if len(set(self.prompt_references)) != len(self.prompt_references):
            raise ValueError("Generation prompts must be unique")

    @property
    def fingerprint(self) -> str:
        """Return a stable SHA-256 fingerprint of the configuration."""

        # Explicit canonical JSON prevents dictionary order and whitespace from
        # changing a cache key across processes or supported platforms.
        payload = {
            "document_content_sha256": self.document_content_sha256,
            "max_batch_chars": self.max_batch_chars,
            "max_map_new_tokens": self.max_map_new_tokens,
            "max_reduce_new_tokens": self.max_reduce_new_tokens,
            "model_name": self.model_name,
            "model_revision": self.model_revision,
            "prompt_references": [
                {
                    "fingerprint": reference.fingerprint,
                    "name": reference.name,
                    "version": reference.version,
                }
                for reference in self.prompt_references
            ],
        }
        canonical_json = json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        return sha256(canonical_json.encode("utf-8")).hexdigest()
