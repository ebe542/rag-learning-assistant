import sqlite3
from contextlib import closing
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path
from uuid import UUID

import pytest

from rag_learning_assistant.generation import Citation
from rag_learning_assistant.learning import (
    QuestionProgress,
    ReviewRating,
    SqliteQuestionProgressRepository,
    SqliteStudyAttemptRepository,
    StudyAttempt,
)

ATTEMPT_ID = UUID("dddddddd-dddd-dddd-dddd-dddddddddddd")
DOCUMENT_ID = UUID("aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa")
BANK_IDENTITY = "b" * 64
ANSWERED_AT = datetime(2026, 8, 14, 12, 0, tzinfo=UTC)


def build_attempt() -> StudyAttempt:
    progress = QuestionProgress(
        document_id=DOCUMENT_ID,
        question_bank_identity_fingerprint=BANK_IDENTITY,
        question_number=1,
        repetition_count=1,
        interval_days=1,
        ease_factor=2.5,
        due_at=ANSWERED_AT + timedelta(days=1),
        last_reviewed_at=ANSWERED_AT,
    )
    return StudyAttempt(
        id=ATTEMPT_ID,
        document_id=DOCUMENT_ID,
        question_bank_identity_fingerprint=BANK_IDENTITY,
        question_number=1,
        question_text="What is retrieval?",
        answer_text="It finds relevant passages.",
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
        rating=ReviewRating.GOOD,
        answered_at=ANSWERED_AT,
        resulting_progress=progress,
    )


def test_study_attempt_survives_repository_reopening(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "metadata.sqlite3"
    attempt = build_attempt()

    SqliteStudyAttemptRepository(database_path).add(attempt)

    reopened = SqliteStudyAttemptRepository(database_path)

    assert reopened.find_by_id(ATTEMPT_ID) == attempt


def test_adding_identical_attempt_twice_is_idempotent(
    tmp_path: Path,
) -> None:
    repository = SqliteStudyAttemptRepository(tmp_path / "metadata.sqlite3")
    attempt = build_attempt()

    repository.add(attempt)
    repository.add(attempt)

    assert repository.find_by_id(attempt.id) == attempt


def test_adding_conflicting_attempt_id_is_rejected(
    tmp_path: Path,
) -> None:
    repository = SqliteStudyAttemptRepository(tmp_path / "metadata.sqlite3")
    attempt = build_attempt()
    repository.add(attempt)

    with pytest.raises(
        ValueError,
        match="Conflicting study attempt already exists",
    ):
        repository.add(
            replace(
                attempt,
                answer_text="A different learner answer.",
            )
        )


def build_attempt_for_document(
    *,
    attempt_id: UUID,
    document_id: UUID,
) -> StudyAttempt:
    attempt = build_attempt()
    progress = replace(
        attempt.resulting_progress,
        document_id=document_id,
    )
    return replace(
        attempt,
        id=attempt_id,
        document_id=document_id,
        resulting_progress=progress,
    )


def test_find_returns_none_for_unknown_attempt(
    tmp_path: Path,
) -> None:
    repository = SqliteStudyAttemptRepository(tmp_path / "metadata.sqlite3")

    assert repository.find_by_id(UUID("eeeeeeee-eeee-eeee-eeee-eeeeeeeeeeee")) is None


def test_delete_document_removes_only_its_attempts(
    tmp_path: Path,
) -> None:
    repository = SqliteStudyAttemptRepository(tmp_path / "metadata.sqlite3")
    other_document_id = UUID("cccccccc-cccc-cccc-cccc-cccccccccccc")
    attempt = build_attempt()
    other_attempt = build_attempt_for_document(
        attempt_id=UUID("eeeeeeee-eeee-eeee-eeee-eeeeeeeeeeee"),
        document_id=other_document_id,
    )
    repository.add(attempt)
    repository.add(other_attempt)

    deleted_count = repository.delete_document(DOCUMENT_ID)

    assert deleted_count == 1
    assert repository.find_by_id(attempt.id) is None
    assert repository.find_by_id(other_attempt.id) == other_attempt


def build_later_attempt() -> StudyAttempt:
    original = build_attempt()
    answered_at = ANSWERED_AT + timedelta(days=1)
    progress = replace(
        original.resulting_progress,
        repetition_count=2,
        interval_days=6,
        due_at=answered_at + timedelta(days=6),
        last_reviewed_at=answered_at,
    )
    return replace(
        original,
        id=UUID("eeeeeeee-eeee-eeee-eeee-eeeeeeeeeeee"),
        answer_text="Retrieval searches the stored source chunks.",
        answered_at=answered_at,
        resulting_progress=progress,
    )


def test_list_question_returns_attempts_in_chronological_order(
    tmp_path: Path,
) -> None:
    repository = SqliteStudyAttemptRepository(tmp_path / "metadata.sqlite3")
    earlier = build_attempt()
    later = build_later_attempt()

    # Insert in reverse order to prove that storage order does not determine
    # the learning history returned to callers.
    repository.add(later)
    repository.add(earlier)

    assert repository.list_question(
        DOCUMENT_ID,
        BANK_IDENTITY,
        1,
    ) == [earlier, later]


def test_list_question_excludes_other_questions(
    tmp_path: Path,
) -> None:
    repository = SqliteStudyAttemptRepository(tmp_path / "metadata.sqlite3")
    attempt = build_attempt()
    other_progress = replace(
        attempt.resulting_progress,
        question_number=2,
    )
    other_question = replace(
        attempt,
        id=UUID("eeeeeeee-eeee-eeee-eeee-eeeeeeeeeeee"),
        question_number=2,
        question_text="What does generation do?",
        resulting_progress=other_progress,
    )
    repository.add(attempt)
    repository.add(other_question)

    assert repository.list_question(
        DOCUMENT_ID,
        BANK_IDENTITY,
        1,
    ) == [attempt]


def test_record_persists_attempt_and_resulting_progress_atomically(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "metadata.sqlite3"
    attempts = SqliteStudyAttemptRepository(database_path)
    progress = SqliteQuestionProgressRepository(database_path)
    attempt = build_attempt()

    attempts.record(attempt)

    assert attempts.find_by_id(attempt.id) == attempt
    assert (
        progress.find(
            attempt.document_id,
            attempt.question_bank_identity_fingerprint,
            attempt.question_number,
        )
        == attempt.resulting_progress
    )


def test_record_initializes_progress_schema_itself(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "metadata.sqlite3"
    repository = SqliteStudyAttemptRepository(database_path)
    attempt = build_attempt()

    repository.record(attempt)

    progress = SqliteQuestionProgressRepository(database_path).find(
        attempt.document_id,
        attempt.question_bank_identity_fingerprint,
        attempt.question_number,
    )

    assert progress == attempt.resulting_progress


def test_failed_attempt_insert_rolls_back_progress_update(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "metadata.sqlite3"
    attempts = SqliteStudyAttemptRepository(database_path)
    progress = SqliteQuestionProgressRepository(database_path)
    attempt = build_attempt()
    original_progress = replace(
        attempt.resulting_progress,
        repetition_count=0,
        interval_days=0,
        due_at=ANSWERED_AT,
        last_reviewed_at=None,
    )
    progress.save(original_progress)

    # The trigger simulates a database failure after record() has updated the
    # progress row but before the transaction can commit the attempt.
    with closing(sqlite3.connect(database_path)) as connection, connection:
        connection.execute(
            """
            CREATE TRIGGER reject_study_attempt
            BEFORE INSERT ON study_attempts
            BEGIN
                SELECT RAISE(ABORT, 'simulated attempt failure');
            END
            """
        )

    with pytest.raises(
        sqlite3.IntegrityError,
        match="simulated attempt failure",
    ):
        attempts.record(attempt)

    assert attempts.find_by_id(attempt.id) is None
    assert (
        progress.find(
            attempt.document_id,
            attempt.question_bank_identity_fingerprint,
            attempt.question_number,
        )
        == original_progress
    )
