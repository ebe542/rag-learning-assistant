from dataclasses import replace
from datetime import UTC, datetime, timedelta
from uuid import UUID

import pytest

from rag_learning_assistant.generation import Citation
from rag_learning_assistant.learning import (
    AnswerEvaluation,
    AnswerVerdict,
    QuestionProgress,
    ReviewRating,
    StudyAttempt,
)

ATTEMPT_ID = UUID("dddddddd-dddd-dddd-dddd-dddddddddddd")
DOCUMENT_ID = UUID("aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa")
BANK_IDENTITY = "b" * 64
ANSWERED_AT = datetime(2026, 8, 14, 12, 0, tzinfo=UTC)


def test_study_attempt_preserves_answer_evidence_and_schedule() -> None:
    citation = Citation(
        number=1,
        source="document.pdf",
        page_number=1,
        chunk_index=0,
        excerpt="Retrieval finds relevant source passages.",
    )
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

    attempt = StudyAttempt(
        id=ATTEMPT_ID,
        document_id=DOCUMENT_ID,
        question_bank_identity_fingerprint=BANK_IDENTITY,
        question_number=1,
        question_text="What is retrieval?",
        answer_text="It finds relevant passages.",
        expected_answer="Retrieval finds relevant source passages.",
        citations=(citation,),
        rating=ReviewRating.GOOD,
        answered_at=ANSWERED_AT,
        resulting_progress=progress,
    )

    assert attempt.id == ATTEMPT_ID
    assert attempt.answer_text == "It finds relevant passages."
    assert attempt.citations == (citation,)
    assert attempt.rating is ReviewRating.GOOD
    assert attempt.resulting_progress == progress


def build_attempt() -> StudyAttempt:
    citation = Citation(
        number=1,
        source="document.pdf",
        page_number=1,
        chunk_index=0,
        excerpt="Retrieval finds relevant source passages.",
    )
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
        citations=(citation,),
        rating=ReviewRating.GOOD,
        answered_at=ANSWERED_AT,
        resulting_progress=progress,
    )


@pytest.mark.parametrize(
    ("field", "message"),
    [
        ("question_text", "Attempt question text must not be blank"),
        ("answer_text", "Attempt answer text must not be blank"),
        (
            "expected_answer",
            "Attempt expected answer must not be blank",
        ),
    ],
)
def test_study_attempt_rejects_blank_text(
    field: str,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        replace(
            build_attempt(),
            **{field: "   "},
        )


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        (
            "question_bank_identity_fingerprint",
            "invalid",
            ("Attempt question bank identity fingerprint must be a lowercase SHA-256 digest"),
        ),
        (
            "question_number",
            0,
            "Attempt question number must be positive",
        ),
        (
            "citations",
            (),
            "Study attempt requires at least one citation",
        ),
    ],
)
def test_study_attempt_rejects_invalid_identity_or_evidence(
    field: str,
    value: object,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        replace(
            build_attempt(),
            **{field: value},
        )


def test_study_attempt_requires_timezone_aware_answer_timestamp() -> None:
    with pytest.raises(
        ValueError,
        match="Attempt answer timestamp must include a timezone",
    ):
        replace(
            build_attempt(),
            answered_at=datetime(2026, 8, 14, 12, 0),
        )


def test_study_attempt_rejects_duplicate_citation_numbers() -> None:
    attempt = build_attempt()
    duplicate = Citation(
        number=attempt.citations[0].number,
        source="document.pdf",
        page_number=2,
        chunk_index=1,
        excerpt="Another supporting passage.",
    )

    with pytest.raises(
        ValueError,
        match="Study attempt citation numbers must be unique",
    ):
        replace(
            attempt,
            citations=(
                attempt.citations[0],
                duplicate,
            ),
        )


@pytest.mark.parametrize(
    ("progress_changes", "message"),
    [
        (
            {"document_id": UUID("cccccccc-cccc-cccc-cccc-cccccccccccc")},
            "Attempt and progress document IDs must match",
        ),
        (
            {
                "question_bank_identity_fingerprint": "c" * 64,
            },
            "Attempt and progress question bank identities must match",
        ),
        (
            {"question_number": 2},
            "Attempt and progress question numbers must match",
        ),
        (
            {"last_reviewed_at": (ANSWERED_AT + timedelta(minutes=1))},
            "Attempt answer timestamp must match progress review timestamp",
        ),
    ],
)
def test_study_attempt_requires_matching_resulting_progress(
    progress_changes: dict[str, object],
    message: str,
) -> None:
    attempt = build_attempt()
    inconsistent_progress = replace(
        attempt.resulting_progress,
        **progress_changes,
    )

    with pytest.raises(ValueError, match=message):
        replace(
            attempt,
            resulting_progress=inconsistent_progress,
        )


def test_study_attempt_preserves_automatic_evaluation() -> None:
    evaluation = AnswerEvaluation(
        verdict=AnswerVerdict.CORRECT,
        score=0.95,
        feedback="The answer covers the expected concept.",
        missing_concepts=(),
    )

    attempt = replace(
        build_attempt(),
        evaluation=evaluation,
    )

    assert attempt.evaluation == evaluation
