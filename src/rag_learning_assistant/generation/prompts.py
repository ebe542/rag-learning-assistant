"""Versioned prompt definitions for reproducible generation."""

from dataclasses import dataclass
from hashlib import sha256


@dataclass(frozen=True, slots=True)
class PromptReference:
    """Compact identity of one exact versioned prompt."""

    name: str
    version: int
    fingerprint: str

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("Prompt reference name must not be blank")

        if self.version < 1:
            raise ValueError("Prompt reference version must be positive")

        is_lowercase_sha256 = len(self.fingerprint) == 64 and all(
            character in "0123456789abcdef" for character in self.fingerprint
        )
        if not is_lowercase_sha256:
            raise ValueError("Prompt fingerprint must be a lowercase SHA-256 hex digest")


@dataclass(frozen=True, slots=True)
class PromptTemplate:
    """Identify an exact prompt text with a human-managed version."""

    name: str
    version: int
    text: str

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("Prompt name must not be blank")

        if self.version < 1:
            raise ValueError("Prompt version must be positive")

        if not self.text.strip():
            raise ValueError("Prompt text must not be blank")

    @property
    def fingerprint(self) -> str:
        """Return the SHA-256 fingerprint of the exact UTF-8 prompt text."""

        # The fingerprint detects accidental text changes even when a developer
        # forgets to increment the human-readable prompt version.
        return sha256(self.text.encode("utf-8")).hexdigest()

    @property
    def reference(self) -> PromptReference:
        """Return the compact identity without exposing the prompt text."""

        return PromptReference(
            name=self.name,
            version=self.version,
            fingerprint=self.fingerprint,
        )
