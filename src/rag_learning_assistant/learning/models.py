import json
from dataclasses import dataclass
from hashlib import sha256
from uuid import UUID

from rag_learning_assistant.generation import Citation, PromptReference


@dataclass(frozen=True, slots=True)
class StudyQuestion:
    """Represent one source-grounded free-response learning question."""

    number: int
    text: str
    expected_answer: str
    citations: tuple[Citation, ...]

    def __post_init__(self) -> None:
        if self.number < 1:
            raise ValueError("Question number must be positive")

        if not self.text.strip():
            raise ValueError("Question text must not be blank")

        if not self.expected_answer.strip():
            raise ValueError(
                "Question expected_answer must not be blank",
            )

        if not self.citations:
            raise ValueError("Question requires at least one citation")

        citation_numbers = [citation.number for citation in self.citations]
        if len(set(citation_numbers)) != len(citation_numbers):
            raise ValueError(
                "Question citation numbers must be unique",
            )


@dataclass(frozen=True, slots=True)
class QuestionBank:
    """Represent one persisted, reproducible question set for a document."""

    document_id: UUID
    identity_fingerprint: str
    source: str
    questions: tuple[StudyQuestion, ...]
    prompt_references: tuple[PromptReference, ...]

    def __post_init__(self) -> None:
        if not self.identity_fingerprint.strip():
            raise ValueError(
                "Question bank identity_fingerprint must not be blank",
            )

        if not self.source.strip():
            raise ValueError("Question bank source must not be blank")

        if not self.questions:
            raise ValueError(
                "Question bank requires at least one question",
            )

        if not self.prompt_references:
            raise ValueError(
                "Question bank requires at least one prompt reference",
            )

        if len(set(self.prompt_references)) != len(self.prompt_references):
            raise ValueError(
                "Question bank prompts must be unique",
            )

        question_numbers = tuple(question.number for question in self.questions)
        expected_numbers = tuple(
            range(1, len(self.questions) + 1),
        )
        if question_numbers != expected_numbers:
            raise ValueError(
                "Question numbers must be consecutive starting at 1",
            )

        if any(
            citation.source != self.source
            for question in self.questions
            for citation in question.citations
        ):
            raise ValueError(
                "Question citation source must match question bank source",
            )

        normalized_questions = [question.text.strip().casefold() for question in self.questions]
        if len(set(normalized_questions)) != len(normalized_questions):
            raise ValueError(
                "Question bank question texts must be unique",
            )


@dataclass(frozen=True, slots=True)
class QuestionBankIdentity:
    """Identify one exact question-bank generation configuration."""

    model_name: str
    model_revision: str
    prompt_references: tuple[PromptReference, ...]
    question_count: int
    max_new_tokens: int
    summary_identity_fingerprint: str

    def __post_init__(self) -> None:
        if not self.model_name.strip() or not self.model_revision.strip():
            raise ValueError(
                "Question bank model fields must not be blank",
            )

        if not self.prompt_references:
            raise ValueError(
                "Question bank identity requires at least one prompt",
            )

        if self.question_count < 1 or self.max_new_tokens < 1:
            raise ValueError(
                "Question bank generation limits must be positive",
            )

        is_summary_sha256 = len(self.summary_identity_fingerprint) == 64 and all(
            character in "0123456789abcdef" for character in self.summary_identity_fingerprint
        )
        if not is_summary_sha256:
            raise ValueError(
                "Summary identity fingerprint must be a lowercase SHA-256 hex digest",
            )

        if len(set(self.prompt_references)) != len(self.prompt_references):
            raise ValueError(
                "Question bank prompts must be unique",
            )

    @property
    def fingerprint(self) -> str:
        """Return a stable SHA-256 fingerprint of the configuration."""

        # Canonical JSON keeps the cache key stable across processes and
        # supported platforms.
        payload = {
            "max_new_tokens": self.max_new_tokens,
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
            "question_count": self.question_count,
            "summary_identity_fingerprint": (self.summary_identity_fingerprint),
        }
        canonical_json = json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        return sha256(
            canonical_json.encode("utf-8"),
        ).hexdigest()
