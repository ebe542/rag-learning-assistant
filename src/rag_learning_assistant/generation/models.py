"""Models for generated answers and their supporting sources."""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Citation:
    """A source passage supporting a generated answer."""

    number: int
    source: str
    page_number: int
    chunk_index: int
    excerpt: str

    def __post_init__(self) -> None:
        if self.number < 1:
            raise ValueError("Citation number must be positive")

        if not self.source.strip():
            raise ValueError("Citation source must not be blank")

        if self.page_number < 1:
            raise ValueError("Citation page number must be positive")

        if self.chunk_index < 0:
            raise ValueError("Citation chunk index must not be negative")

        if not self.excerpt.strip():
            raise ValueError("Citation excerpt must not be blank")


@dataclass(frozen=True, slots=True)
class GroundedAnswer:
    """An answer together with the question and cited sources."""

    question: str
    text: str
    citations: tuple[Citation, ...]

    def __post_init__(self) -> None:
        if not self.question.strip():
            raise ValueError("Answer question must not be blank")

        if not self.text.strip():
            raise ValueError("Answer text must not be blank")


@dataclass(frozen=True, slots=True)
class GenerationResult:
    """Raw model output with references to numbered prompt contexts."""

    text: str
    citation_numbers: tuple[int, ...]

    def __post_init__(self) -> None:
        if not self.text.strip():
            raise ValueError("Generated text must not be blank")

        if any(number < 1 for number in self.citation_numbers):
            raise ValueError("Citation numbers must be positive")

        if len(set(self.citation_numbers)) != len(self.citation_numbers):
            raise ValueError("Citation numbers must be unique")
