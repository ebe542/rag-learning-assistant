import json
from datetime import UTC, datetime
from pathlib import Path
from uuid import UUID

from rag_learning_assistant.application import (
    DueQuestion,
    QuestionBankCatalog,
    ReviewService,
)
from rag_learning_assistant.generation import Citation
from rag_learning_assistant.interfaces.cli import commands, entrypoint
from rag_learning_assistant.interfaces.cli.parser import (
    DEFAULT_REVIEW_LIMIT,
    build_parser,
)
from rag_learning_assistant.learning import (
    QuestionProgress,
    ReviewRating,
    SqliteQuestionProgressRepository,
    StudyQuestion,
)

DOCUMENT_ID = UUID("aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa")
BANK_IDENTITY = "b" * 64


def test_parser_accepts_review_due_command() -> None:
    args = build_parser().parse_args(
        [
            "review-due",
            "local-data/indexes/library",
            str(DOCUMENT_ID),
            BANK_IDENTITY,
        ]
    )

    assert args.command == "review-due"
    assert args.index_dir == Path("local-data/indexes/library")
    assert args.document_id == DOCUMENT_ID
    assert args.question_bank_identity_fingerprint == BANK_IDENTITY
    assert args.limit == DEFAULT_REVIEW_LIMIT


def test_parser_accepts_review_record_command() -> None:
    args = build_parser().parse_args(
        [
            "review-record",
            "local-data/indexes/library",
            str(DOCUMENT_ID),
            BANK_IDENTITY,
            "2",
            "good",
        ]
    )

    assert args.command == "review-record"
    assert args.index_dir == Path("local-data/indexes/library")
    assert args.document_id == DOCUMENT_ID
    assert args.question_bank_identity_fingerprint == BANK_IDENTITY
    assert args.question_number == 2
    assert args.rating is ReviewRating.GOOD


def test_review_service_builder_uses_persistent_library_storage(
    tmp_path: Path,
) -> None:
    service = commands.build_review_service(tmp_path)

    assert isinstance(service, ReviewService)
    assert isinstance(service.banks, QuestionBankCatalog)
    assert isinstance(
        service.progress,
        SqliteQuestionProgressRepository,
    )
    assert service.progress.database_path == (tmp_path / "metadata.sqlite3")


AS_OF = datetime(2026, 8, 14, 12, 0, tzinfo=UTC)


class RecordingReviewService:
    def __init__(
        self,
        due: list[DueQuestion],
        reviewed: QuestionProgress | None = None,
    ) -> None:
        self.due = due
        self.reviewed = reviewed
        self.list_due_calls: list[tuple[UUID, str, datetime, int]] = []
        self.record_review_calls: list[tuple[UUID, str, int, ReviewRating, datetime]] = []

    def list_due(
        self,
        document_id: UUID,
        question_bank_identity_fingerprint: str,
        *,
        as_of: datetime,
        limit: int,
    ) -> list[DueQuestion]:
        self.list_due_calls.append(
            (
                document_id,
                question_bank_identity_fingerprint,
                as_of,
                limit,
            )
        )
        return self.due

    def record_review(
        self,
        document_id: UUID,
        question_bank_identity_fingerprint: str,
        question_number: int,
        rating: ReviewRating,
        *,
        reviewed_at: datetime,
    ) -> QuestionProgress:
        self.record_review_calls.append(
            (
                document_id,
                question_bank_identity_fingerprint,
                question_number,
                rating,
                reviewed_at,
            )
        )
        assert self.reviewed is not None
        return self.reviewed


def test_run_review_due_outputs_new_questions_as_json(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    question = StudyQuestion(
        number=1,
        text="What is retrieval?",
        expected_answer="Retrieval finds relevant source passages.",
        citations=(
            Citation(
                number=1,
                source="document.pdf",
                page_number=1,
                chunk_index=0,
                excerpt="Retrieval finds relevant source passages.",
            ),
        ),
    )
    service = RecordingReviewService(
        due=[
            DueQuestion(
                question=question,
                progress=None,
            )
        ]
    )
    monkeypatch.setattr(
        commands,
        "build_review_service",
        lambda index_directory: service,
    )

    exit_code = commands.run_review_due(
        tmp_path,
        DOCUMENT_ID,
        BANK_IDENTITY,
        10,
        as_of=AS_OF,
    )

    assert exit_code == 0
    assert service.list_due_calls == [
        (DOCUMENT_ID, BANK_IDENTITY, AS_OF, 10),
    ]
    assert json.loads(capsys.readouterr().out) == {
        "index_directory": str(tmp_path),
        "document_id": str(DOCUMENT_ID),
        "question_bank_identity_fingerprint": BANK_IDENTITY,
        "as_of": AS_OF.isoformat(),
        "questions": [
            {
                "number": 1,
                "text": "What is retrieval?",
                "expected_answer": ("Retrieval finds relevant source passages."),
                "citations": [
                    {
                        "number": 1,
                        "source": "document.pdf",
                        "page_number": 1,
                        "chunk_index": 0,
                        "excerpt": ("Retrieval finds relevant source passages."),
                    }
                ],
                "progress": None,
            }
        ],
    }


def test_run_review_record_outputs_updated_schedule_as_json(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    reviewed = QuestionProgress(
        document_id=DOCUMENT_ID,
        question_bank_identity_fingerprint=BANK_IDENTITY,
        question_number=1,
        repetition_count=1,
        interval_days=1,
        ease_factor=2.5,
        due_at=AS_OF.replace(day=15),
        last_reviewed_at=AS_OF,
    )
    service = RecordingReviewService(
        due=[],
        reviewed=reviewed,
    )
    monkeypatch.setattr(
        commands,
        "build_review_service",
        lambda index_directory: service,
    )

    exit_code = commands.run_review_record(
        tmp_path,
        DOCUMENT_ID,
        BANK_IDENTITY,
        1,
        ReviewRating.GOOD,
        reviewed_at=AS_OF,
    )

    assert exit_code == 0
    assert service.record_review_calls == [
        (
            DOCUMENT_ID,
            BANK_IDENTITY,
            1,
            ReviewRating.GOOD,
            AS_OF,
        )
    ]
    assert json.loads(capsys.readouterr().out) == {
        "index_directory": str(tmp_path),
        "document_id": str(DOCUMENT_ID),
        "question_bank_identity_fingerprint": BANK_IDENTITY,
        "question_number": 1,
        "rating": "good",
        "progress": {
            "repetition_count": 1,
            "interval_days": 1,
            "ease_factor": 2.5,
            "due_at": reviewed.due_at.isoformat(),
            "last_reviewed_at": AS_OF.isoformat(),
        },
    }


def test_entrypoint_dispatches_review_due_command(
    tmp_path: Path,
    monkeypatch,
) -> None:
    (tmp_path / "metadata.sqlite3").write_bytes(b"")
    calls: list[tuple[Path, UUID, str, int]] = []

    def fake_run_review_due(
        index_directory: Path,
        document_id: UUID,
        question_bank_identity_fingerprint: str,
        limit: int,
    ) -> int:
        calls.append(
            (
                index_directory,
                document_id,
                question_bank_identity_fingerprint,
                limit,
            )
        )
        return 0

    monkeypatch.setattr(
        commands,
        "run_review_due",
        fake_run_review_due,
    )

    exit_code = entrypoint.main(
        [
            "review-due",
            str(tmp_path),
            str(DOCUMENT_ID),
            BANK_IDENTITY,
            "--limit",
            "3",
        ]
    )

    assert exit_code == 0
    assert calls == [
        (tmp_path, DOCUMENT_ID, BANK_IDENTITY, 3),
    ]


def test_entrypoint_dispatches_review_record_command(
    tmp_path: Path,
    monkeypatch,
) -> None:
    (tmp_path / "metadata.sqlite3").write_bytes(b"")
    calls: list[tuple[Path, UUID, str, int, ReviewRating]] = []

    def fake_run_review_record(
        index_directory: Path,
        document_id: UUID,
        question_bank_identity_fingerprint: str,
        question_number: int,
        rating: ReviewRating,
    ) -> int:
        calls.append(
            (
                index_directory,
                document_id,
                question_bank_identity_fingerprint,
                question_number,
                rating,
            )
        )
        return 0

    monkeypatch.setattr(
        commands,
        "run_review_record",
        fake_run_review_record,
    )

    exit_code = entrypoint.main(
        [
            "review-record",
            str(tmp_path),
            str(DOCUMENT_ID),
            BANK_IDENTITY,
            "2",
            "hard",
        ]
    )

    assert exit_code == 0
    assert calls == [
        (
            tmp_path,
            DOCUMENT_ID,
            BANK_IDENTITY,
            2,
            ReviewRating.HARD,
        )
    ]
