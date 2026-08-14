from datetime import UTC, datetime, timedelta
from pathlib import Path
from uuid import UUID

from rag_learning_assistant.learning import (
    QuestionProgress,
    SqliteQuestionProgressRepository,
)

DOCUMENT_ID = UUID("aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa")
BANK_IDENTITY = "b" * 64
REVIEWED_AT = datetime(2026, 8, 14, 12, 0, tzinfo=UTC)


def build_progress() -> QuestionProgress:
    return QuestionProgress(
        document_id=DOCUMENT_ID,
        question_bank_identity_fingerprint=BANK_IDENTITY,
        question_number=1,
        repetition_count=2,
        interval_days=6,
        ease_factor=2.5,
        due_at=REVIEWED_AT + timedelta(days=6),
        last_reviewed_at=REVIEWED_AT,
    )


def test_question_progress_survives_repository_reopening(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "metadata.sqlite3"
    progress = build_progress()

    SqliteQuestionProgressRepository(database_path).save(progress)

    reopened = SqliteQuestionProgressRepository(database_path)

    assert (
        reopened.find(
            DOCUMENT_ID,
            BANK_IDENTITY,
            1,
        )
        == progress
    )


def test_saving_progress_replaces_the_previous_schedule(
    tmp_path: Path,
) -> None:
    repository = SqliteQuestionProgressRepository(tmp_path / "metadata.sqlite3")
    original = build_progress()
    updated = QuestionProgress(
        document_id=original.document_id,
        question_bank_identity_fingerprint=(original.question_bank_identity_fingerprint),
        question_number=original.question_number,
        repetition_count=3,
        interval_days=15,
        ease_factor=2.5,
        due_at=REVIEWED_AT + timedelta(days=15),
        last_reviewed_at=REVIEWED_AT + timedelta(days=6),
    )

    repository.save(original)
    repository.save(updated)

    assert (
        repository.find(
            DOCUMENT_ID,
            BANK_IDENTITY,
            1,
        )
        == updated
    )


def test_find_returns_none_for_unknown_question(
    tmp_path: Path,
) -> None:
    repository = SqliteQuestionProgressRepository(tmp_path / "metadata.sqlite3")

    assert (
        repository.find(
            DOCUMENT_ID,
            BANK_IDENTITY,
            99,
        )
        is None
    )


def test_delete_document_removes_only_its_question_progress(
    tmp_path: Path,
) -> None:
    repository = SqliteQuestionProgressRepository(tmp_path / "metadata.sqlite3")
    other_document_id = UUID("cccccccc-cccc-cccc-cccc-cccccccccccc")
    progress = build_progress()
    other_progress = QuestionProgress(
        document_id=other_document_id,
        question_bank_identity_fingerprint=BANK_IDENTITY,
        question_number=1,
        repetition_count=1,
        interval_days=1,
        ease_factor=2.5,
        due_at=REVIEWED_AT + timedelta(days=1),
        last_reviewed_at=REVIEWED_AT,
    )
    repository.save(progress)
    repository.save(other_progress)

    deleted_count = repository.delete_document(DOCUMENT_ID)

    assert deleted_count == 1
    assert repository.find(DOCUMENT_ID, BANK_IDENTITY, 1) is None
    assert (
        repository.find(
            other_document_id,
            BANK_IDENTITY,
            1,
        )
        == other_progress
    )
