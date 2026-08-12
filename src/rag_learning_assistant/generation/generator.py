"""Provider-independent interface for text generation."""

from typing import Protocol

from rag_learning_assistant.generation.models import GenerationResult


class TextGenerator(Protocol):
    """Generate an answer from a fully prepared prompt."""

    def generate(
        self,
        prompt: str,
        *,
        max_new_tokens: int | None = None,
    ) -> GenerationResult:
        """Return generated text and referenced context numbers."""
        ...
