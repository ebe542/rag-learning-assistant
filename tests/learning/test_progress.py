from dataclasses import FrozenInstanceError
from datetime import UTC, datetime
from uuid import UUID

import pytest

from rag_learning_assistant.learning import QuestionProgress, ReviewRating

DOCUMENT_ID = UUID("aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa")
BANK_IDENTITY = "b" * 64


def test_question_progress_represents_an_unreviewed_question() -> None:
    due_at = datetime(2026, 8, 14, 12, 0, tzinfo=UTC)

    progress = QuestionProgress(
        document_id=DOCUMENT_ID,
        question_bank_identity_fingerprint=BANK_IDENTITY,
        question_number=1,
        repetition_count=0,
        interval_days=0,
        ease_factor=2.5,
        due_at=due_at,
        last_reviewed_at=None,
    )

    assert progress.document_id == DOCUMENT_ID
    assert progress.question_bank_identity_fingerprint == BANK_IDENTITY
    assert progress.question_number == 1
    assert progress.repetition_count == 0
    assert progress.interval_days == 0
    assert progress.ease_factor == 2.5
    assert progress.due_at == due_at
    assert progress.last_reviewed_at is None


def test_question_progress_is_immutable() -> None:
    progress = QuestionProgress(
        document_id=DOCUMENT_ID,
        question_bank_identity_fingerprint=BANK_IDENTITY,
        question_number=1,
        repetition_count=0,
        interval_days=0,
        ease_factor=2.5,
        due_at=datetime(2026, 8, 14, 12, 0, tzinfo=UTC),
        last_reviewed_at=None,
    )

    with pytest.raises(FrozenInstanceError):
        progress.repetition_count = 1  # type: ignore[misc]


def test_review_rating_has_stable_storage_values() -> None:
    assert ReviewRating.AGAIN.value == "again"
    assert ReviewRating.HARD.value == "hard"
    assert ReviewRating.GOOD.value == "good"
    assert ReviewRating.EASY.value == "easy"


@pytest.mark.parametrize(
    ("fingerprint", "message"),
    [
        ("", "Question bank identity fingerprint must be a lowercase SHA-256 digest"),
        ("b" * 63, "Question bank identity fingerprint must be a lowercase SHA-256 digest"),
        ("B" * 64, "Question bank identity fingerprint must be a lowercase SHA-256 digest"),
        ("z" * 64, "Question bank identity fingerprint must be a lowercase SHA-256 digest"),
    ],
)
def test_question_progress_requires_valid_bank_identity(
    fingerprint: str,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        QuestionProgress(
            document_id=DOCUMENT_ID,
            question_bank_identity_fingerprint=fingerprint,
            question_number=1,
            repetition_count=0,
            interval_days=0,
            ease_factor=2.5,
            due_at=datetime(2026, 8, 14, 12, 0, tzinfo=UTC),
            last_reviewed_at=None,
        )


@pytest.mark.parametrize("question_number", [0, -1])
def test_question_progress_requires_positive_question_number(
    question_number: int,
) -> None:
    with pytest.raises(
        ValueError,
        match="Question number must be positive",
    ):
        QuestionProgress(
            document_id=DOCUMENT_ID,
            question_bank_identity_fingerprint=BANK_IDENTITY,
            question_number=question_number,
            repetition_count=0,
            interval_days=0,
            ease_factor=2.5,
            due_at=datetime(2026, 8, 14, 12, 0, tzinfo=UTC),
            last_reviewed_at=None,
        )


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("repetition_count", -1, "Repetition count must not be negative"),
        ("interval_days", -1, "Review interval must not be negative"),
        ("ease_factor", 1.29, "Ease factor must be at least 1.3"),
    ],
)
def test_question_progress_rejects_invalid_schedule_numbers(
    field: str,
    value: int | float,
    message: str,
) -> None:
    values = {
        "document_id": DOCUMENT_ID,
        "question_bank_identity_fingerprint": BANK_IDENTITY,
        "question_number": 1,
        "repetition_count": 0,
        "interval_days": 0,
        "ease_factor": 2.5,
        "due_at": datetime(2026, 8, 14, 12, 0, tzinfo=UTC),
        "last_reviewed_at": None,
    }
    values[field] = value

    with pytest.raises(ValueError, match=message):
        QuestionProgress(**values)  # type: ignore[arg-type]


def test_question_progress_requires_timezone_aware_due_date() -> None:
    with pytest.raises(
        ValueError,
        match="Due date must include a timezone",
    ):
        QuestionProgress(
            document_id=DOCUMENT_ID,
            question_bank_identity_fingerprint=BANK_IDENTITY,
            question_number=1,
            repetition_count=0,
            interval_days=0,
            ease_factor=2.5,
            due_at=datetime(2026, 8, 14, 12, 0),
            last_reviewed_at=None,
        )


def test_question_progress_requires_timezone_aware_last_review_date() -> None:
    with pytest.raises(
        ValueError,
        match="Last review date must include a timezone",
    ):
        QuestionProgress(
            document_id=DOCUMENT_ID,
            question_bank_identity_fingerprint=BANK_IDENTITY,
            question_number=1,
            repetition_count=1,
            interval_days=1,
            ease_factor=2.5,
            due_at=datetime(2026, 8, 15, 12, 0, tzinfo=UTC),
            last_reviewed_at=datetime(2026, 8, 14, 12, 0),
        )


def test_question_progress_rejects_due_date_before_last_review() -> None:
    with pytest.raises(
        ValueError,
        match="Due date must not be earlier than the last review date",
    ):
        QuestionProgress(
            document_id=DOCUMENT_ID,
            question_bank_identity_fingerprint=BANK_IDENTITY,
            question_number=1,
            repetition_count=1,
            interval_days=1,
            ease_factor=2.5,
            due_at=datetime(2026, 8, 14, 12, 0, tzinfo=UTC),
            last_reviewed_at=datetime(2026, 8, 15, 12, 0, tzinfo=UTC),
        )
