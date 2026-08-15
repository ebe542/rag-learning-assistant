from datetime import UTC, datetime, timedelta
from pathlib import Path
from uuid import UUID

import pytest

from rag_learning_assistant.application import (
    DueQuestion,
    QuestionBankCatalog,
    ReviewService,
    StudySessionService,
)
from rag_learning_assistant.generation import Citation
from rag_learning_assistant.interfaces.cli import commands, entrypoint
from rag_learning_assistant.interfaces.cli.parser import build_parser
from rag_learning_assistant.interfaces.cli.study import (
    conduct_study_question,
)
from rag_learning_assistant.learning import (
    QuestionProgress,
    ReviewRating,
    SqliteStudyAttemptRepository,
    StudyAttempt,
    StudyQuestion,
)

DOCUMENT_ID = UUID("aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa")
BANK_IDENTITY = "b" * 64
AS_OF = datetime(2026, 8, 14, 12, 0, tzinfo=UTC)


def test_parser_accepts_interactive_study_command() -> None:
    args = build_parser().parse_args(
        [
            "study",
            "library-index",
            str(DOCUMENT_ID),
            BANK_IDENTITY,
        ]
    )

    assert args.command == "study"
    assert args.index_dir == Path("library-index")
    assert args.document_id == DOCUMENT_ID
    assert args.question_bank_identity_fingerprint == BANK_IDENTITY


def test_conduct_study_question_reveals_answer_before_rating() -> None:
    question = StudyQuestion(
        number=1,
        text="What is retrieval?",
        expected_answer="Retrieval finds relevant source passages.",
        citations=(
            Citation(
                number=1,
                source="document.pdf",
                page_number=3,
                chunk_index=4,
                excerpt="Retrieval finds relevant source passages.",
            ),
        ),
    )
    events: list[tuple[str, str]] = []
    answers = iter(
        [
            "It finds relevant passages.",
            "good",
        ]
    )

    def read_line(prompt: str) -> str:
        events.append(("read", prompt))
        return next(answers)

    def write_line(text: str) -> None:
        events.append(("write", text))

    answer_text, rating = conduct_study_question(
        question,
        read_line=read_line,
        write_line=write_line,
    )

    assert answer_text == "It finds relevant passages."
    assert rating is ReviewRating.GOOD
    assert events == [
        ("write", "Question 1: What is retrieval?"),
        ("read", "Your answer: "),
        (
            "write",
            "Expected answer: Retrieval finds relevant source passages.",
        ),
        (
            "write",
            "Source 1: document.pdf, page 3, chunk 4",
        ),
        ("read", "Rating [again/hard/good/easy]: "),
    ]


def test_conduct_study_question_repeats_invalid_inputs() -> None:
    question = StudyQuestion(
        number=1,
        text="What is retrieval?",
        expected_answer="Retrieval finds relevant source passages.",
        citations=(
            Citation(
                number=1,
                source="document.pdf",
                page_number=3,
                chunk_index=4,
                excerpt="Retrieval finds relevant source passages.",
            ),
        ),
    )
    inputs = iter(
        [
            "   ",
            "It finds relevant passages.",
            "unknown",
            "hard",
        ]
    )
    output: list[str] = []

    answer_text, rating = conduct_study_question(
        question,
        read_line=lambda prompt: next(inputs),
        write_line=output.append,
    )

    assert answer_text == "It finds relevant passages."
    assert rating is ReviewRating.HARD
    assert "Answer must not be blank." in output
    assert "Rating must be again, hard, good, or easy." in output


def test_entrypoint_dispatches_study_command(
    tmp_path: Path,
    monkeypatch,
) -> None:
    (tmp_path / "metadata.sqlite3").write_bytes(b"")
    calls: list[tuple[Path, UUID, str]] = []

    def fake_run_study(
        index_directory: Path,
        document_id: UUID,
        question_bank_identity_fingerprint: str,
    ) -> int:
        calls.append(
            (
                index_directory,
                document_id,
                question_bank_identity_fingerprint,
            )
        )
        return 0

    monkeypatch.setattr(
        commands,
        "run_study",
        fake_run_study,
    )

    exit_code = entrypoint.main(
        [
            "study",
            str(tmp_path),
            str(DOCUMENT_ID),
            BANK_IDENTITY,
        ]
    )

    assert exit_code == 0
    assert calls == [
        (
            tmp_path,
            DOCUMENT_ID,
            BANK_IDENTITY,
        )
    ]


def test_study_session_builder_uses_persistent_library_storage(
    tmp_path: Path,
) -> None:
    service = commands.build_study_session_service(tmp_path)

    assert isinstance(service, StudySessionService)
    assert isinstance(service.banks, QuestionBankCatalog)
    assert isinstance(service.reviewer, ReviewService)
    assert isinstance(
        service.attempts,
        SqliteStudyAttemptRepository,
    )
    assert service.attempts.database_path == (tmp_path / "metadata.sqlite3")


class EmptyStudySessionService:
    def __init__(self) -> None:
        self.next_due_calls: list[tuple[UUID, str, datetime]] = []

    def next_due(
        self,
        document_id: UUID,
        question_bank_identity_fingerprint: str,
        *,
        as_of: datetime,
    ):
        self.next_due_calls.append(
            (
                document_id,
                question_bank_identity_fingerprint,
                as_of,
            )
        )
        return None


def test_run_study_reports_when_no_question_is_due(
    tmp_path: Path,
    monkeypatch,
) -> None:
    service = EmptyStudySessionService()
    output: list[str] = []

    monkeypatch.setattr(
        commands,
        "build_study_session_service",
        lambda index_directory: service,
    )

    exit_code = commands.run_study(
        tmp_path,
        DOCUMENT_ID,
        BANK_IDENTITY,
        as_of=AS_OF,
        read_line=lambda prompt: pytest.fail("Input must not be requested without a due question"),
        write_line=output.append,
    )

    assert exit_code == 0
    assert service.next_due_calls == [
        (
            DOCUMENT_ID,
            BANK_IDENTITY,
            AS_OF,
        )
    ]
    assert output == ["No study questions are due."]


class RecordingStudySessionService:
    def __init__(
        self,
        due: DueQuestion,
        attempt: StudyAttempt,
    ) -> None:
        self.due = due
        self.attempt = attempt
        self.next_due_calls: list[tuple[UUID, str, datetime]] = []
        self.record_answer_calls: list[tuple[UUID, str, int, str, ReviewRating, datetime]] = []

    def next_due(
        self,
        document_id: UUID,
        question_bank_identity_fingerprint: str,
        *,
        as_of: datetime,
    ) -> DueQuestion:
        self.next_due_calls.append(
            (
                document_id,
                question_bank_identity_fingerprint,
                as_of,
            )
        )
        return self.due

    def record_answer(
        self,
        document_id: UUID,
        question_bank_identity_fingerprint: str,
        question_number: int,
        *,
        answer_text: str,
        rating: ReviewRating,
        answered_at: datetime,
    ) -> StudyAttempt:
        self.record_answer_calls.append(
            (
                document_id,
                question_bank_identity_fingerprint,
                question_number,
                answer_text,
                rating,
                answered_at,
            )
        )
        return self.attempt


def test_run_study_records_interactive_answer(
    tmp_path: Path,
    monkeypatch,
) -> None:
    question = StudyQuestion(
        number=1,
        text="What is retrieval?",
        expected_answer="Retrieval finds relevant source passages.",
        citations=(
            Citation(
                number=1,
                source="document.pdf",
                page_number=3,
                chunk_index=4,
                excerpt="Retrieval finds relevant source passages.",
            ),
        ),
    )
    progress = QuestionProgress(
        document_id=DOCUMENT_ID,
        question_bank_identity_fingerprint=BANK_IDENTITY,
        question_number=1,
        repetition_count=1,
        interval_days=1,
        ease_factor=2.5,
        due_at=AS_OF + timedelta(days=1),
        last_reviewed_at=AS_OF,
    )
    attempt = StudyAttempt(
        id=UUID("dddddddd-dddd-dddd-dddd-dddddddddddd"),
        document_id=DOCUMENT_ID,
        question_bank_identity_fingerprint=BANK_IDENTITY,
        question_number=1,
        question_text=question.text,
        answer_text="It finds relevant passages.",
        expected_answer=question.expected_answer,
        citations=question.citations,
        rating=ReviewRating.GOOD,
        answered_at=AS_OF,
        resulting_progress=progress,
    )
    service = RecordingStudySessionService(
        due=DueQuestion(question=question, progress=None),
        attempt=attempt,
    )
    inputs = iter(["It finds relevant passages.", "good"])
    output: list[str] = []

    monkeypatch.setattr(
        commands,
        "build_study_session_service",
        lambda index_directory: service,
    )

    exit_code = commands.run_study(
        tmp_path,
        DOCUMENT_ID,
        BANK_IDENTITY,
        as_of=AS_OF,
        read_line=lambda prompt: next(inputs),
        write_line=output.append,
    )

    assert exit_code == 0
    assert service.record_answer_calls == [
        (
            DOCUMENT_ID,
            BANK_IDENTITY,
            1,
            "It finds relevant passages.",
            ReviewRating.GOOD,
            AS_OF,
        )
    ]
    assert output[-1] == (f"Review recorded. Next due: {progress.due_at.isoformat()}")
