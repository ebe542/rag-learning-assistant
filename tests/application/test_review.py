from datetime import UTC, datetime, timedelta
from uuid import UUID

import pytest

from rag_learning_assistant.application import (
    DueQuestion,
    ReviewScheduler,
    ReviewService,
    StudyQuestionNotFoundError,
)
from rag_learning_assistant.generation import Citation, PromptReference
from rag_learning_assistant.learning import (
    QuestionBank,
    QuestionProgress,
    ReviewRating,
    StudyQuestion,
)

DOCUMENT_ID = UUID("aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa")
BANK_IDENTITY = "b" * 64
REVIEWED_AT = datetime(2026, 8, 14, 12, 0, tzinfo=UTC)


def test_good_rating_schedules_new_question_for_next_day() -> None:
    progress = QuestionProgress(
        document_id=DOCUMENT_ID,
        question_bank_identity_fingerprint=BANK_IDENTITY,
        question_number=1,
        repetition_count=0,
        interval_days=0,
        ease_factor=2.5,
        due_at=REVIEWED_AT,
        last_reviewed_at=None,
    )

    reviewed = ReviewScheduler().review(
        progress,
        ReviewRating.GOOD,
        reviewed_at=REVIEWED_AT,
    )

    assert reviewed == QuestionProgress(
        document_id=DOCUMENT_ID,
        question_bank_identity_fingerprint=BANK_IDENTITY,
        question_number=1,
        repetition_count=1,
        interval_days=1,
        ease_factor=2.5,
        due_at=REVIEWED_AT + timedelta(days=1),
        last_reviewed_at=REVIEWED_AT,
    )


@pytest.mark.parametrize(
    (
        "rating",
        "expected_repetitions",
        "expected_interval_days",
        "expected_ease_factor",
        "expected_due_at",
    ),
    [
        (
            ReviewRating.AGAIN,
            0,
            0,
            2.3,
            REVIEWED_AT + timedelta(minutes=10),
        ),
        (
            ReviewRating.HARD,
            1,
            1,
            2.35,
            REVIEWED_AT + timedelta(days=1),
        ),
        (
            ReviewRating.EASY,
            1,
            4,
            2.65,
            REVIEWED_AT + timedelta(days=4),
        ),
    ],
)
def test_rating_schedules_a_new_question(
    rating: ReviewRating,
    expected_repetitions: int,
    expected_interval_days: int,
    expected_ease_factor: float,
    expected_due_at: datetime,
) -> None:
    progress = QuestionProgress(
        document_id=DOCUMENT_ID,
        question_bank_identity_fingerprint=BANK_IDENTITY,
        question_number=1,
        repetition_count=0,
        interval_days=0,
        ease_factor=2.5,
        due_at=REVIEWED_AT,
        last_reviewed_at=None,
    )

    reviewed = ReviewScheduler().review(
        progress,
        rating,
        reviewed_at=REVIEWED_AT,
    )

    assert reviewed.repetition_count == expected_repetitions
    assert reviewed.interval_days == expected_interval_days
    assert reviewed.ease_factor == pytest.approx(expected_ease_factor)
    assert reviewed.due_at == expected_due_at
    assert reviewed.last_reviewed_at == REVIEWED_AT


def test_second_good_rating_uses_six_day_interval() -> None:
    progress = QuestionProgress(
        document_id=DOCUMENT_ID,
        question_bank_identity_fingerprint=BANK_IDENTITY,
        question_number=1,
        repetition_count=1,
        interval_days=1,
        ease_factor=2.5,
        due_at=REVIEWED_AT,
        last_reviewed_at=REVIEWED_AT - timedelta(days=1),
    )

    reviewed = ReviewScheduler().review(
        progress,
        ReviewRating.GOOD,
        reviewed_at=REVIEWED_AT,
    )

    assert reviewed.repetition_count == 2
    assert reviewed.interval_days == 6
    assert reviewed.due_at == REVIEWED_AT + timedelta(days=6)


@pytest.mark.parametrize(
    ("rating", "expected_repetitions", "expected_interval", "expected_ease"),
    [
        (ReviewRating.AGAIN, 0, 0, 2.3),
        (ReviewRating.HARD, 3, 7, 2.35),
        (ReviewRating.GOOD, 3, 15, 2.5),
        (ReviewRating.EASY, 3, 20, 2.65),
    ],
)
def test_rating_updates_a_mature_question(
    rating: ReviewRating,
    expected_repetitions: int,
    expected_interval: int,
    expected_ease: float,
) -> None:
    progress = QuestionProgress(
        document_id=DOCUMENT_ID,
        question_bank_identity_fingerprint=BANK_IDENTITY,
        question_number=1,
        repetition_count=2,
        interval_days=6,
        ease_factor=2.5,
        due_at=REVIEWED_AT,
        last_reviewed_at=REVIEWED_AT - timedelta(days=6),
    )

    reviewed = ReviewScheduler().review(
        progress,
        rating,
        reviewed_at=REVIEWED_AT,
    )

    assert reviewed.repetition_count == expected_repetitions
    assert reviewed.interval_days == expected_interval
    assert reviewed.ease_factor == pytest.approx(expected_ease)

    expected_due_at = (
        REVIEWED_AT + timedelta(minutes=10)
        if rating is ReviewRating.AGAIN
        else REVIEWED_AT + timedelta(days=expected_interval)
    )
    assert reviewed.due_at == expected_due_at


def test_review_requires_timezone_aware_timestamp() -> None:
    progress = QuestionProgress(
        document_id=DOCUMENT_ID,
        question_bank_identity_fingerprint=BANK_IDENTITY,
        question_number=1,
        repetition_count=0,
        interval_days=0,
        ease_factor=2.5,
        due_at=REVIEWED_AT,
        last_reviewed_at=None,
    )

    with pytest.raises(
        ValueError,
        match="Review timestamp must include a timezone",
    ):
        ReviewScheduler().review(
            progress,
            ReviewRating.GOOD,
            reviewed_at=datetime(2026, 8, 14, 12, 0),
        )


def test_review_cannot_precede_the_previous_review() -> None:
    progress = QuestionProgress(
        document_id=DOCUMENT_ID,
        question_bank_identity_fingerprint=BANK_IDENTITY,
        question_number=1,
        repetition_count=1,
        interval_days=1,
        ease_factor=2.5,
        due_at=REVIEWED_AT + timedelta(days=1),
        last_reviewed_at=REVIEWED_AT,
    )

    with pytest.raises(
        ValueError,
        match="Review timestamp must not precede the previous review",
    ):
        ReviewScheduler().review(
            progress,
            ReviewRating.GOOD,
            reviewed_at=REVIEWED_AT - timedelta(minutes=1),
        )


def build_question_bank() -> QuestionBank:
    return QuestionBank(
        document_id=DOCUMENT_ID,
        identity_fingerprint=BANK_IDENTITY,
        source="document.pdf",
        questions=(
            StudyQuestion(
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
            ),
        ),
        prompt_references=(
            PromptReference(
                name="question-bank.generate",
                version=1,
                fingerprint="c" * 64,
            ),
        ),
    )


class StaticQuestionBankCatalog:
    def __init__(self, bank: QuestionBank) -> None:
        self.bank = bank
        self.calls: list[tuple[UUID, str]] = []

    def get_document_bank(
        self,
        document_id: UUID,
        identity_fingerprint: str,
    ) -> QuestionBank:
        self.calls.append((document_id, identity_fingerprint))
        return self.bank


class RecordingProgressRepository:
    def __init__(
        self,
        found: QuestionProgress | None = None,
    ) -> None:
        self.found = found
        self.find_calls: list[tuple[UUID, str, int]] = []
        self.saved: list[QuestionProgress] = []

    def find(
        self,
        document_id: UUID,
        question_bank_identity_fingerprint: str,
        question_number: int,
    ) -> QuestionProgress | None:
        self.find_calls.append(
            (
                document_id,
                question_bank_identity_fingerprint,
                question_number,
            )
        )
        return self.found

    def save(self, progress: QuestionProgress) -> None:
        self.saved.append(progress)


def test_review_service_creates_and_saves_progress_for_new_question() -> None:
    bank = build_question_bank()
    banks = StaticQuestionBankCatalog(bank)
    progress = RecordingProgressRepository()
    service = ReviewService(
        banks=banks,
        progress=progress,
        scheduler=ReviewScheduler(),
    )

    reviewed = service.record_review(
        DOCUMENT_ID,
        BANK_IDENTITY,
        1,
        ReviewRating.GOOD,
        reviewed_at=REVIEWED_AT,
    )

    assert reviewed.repetition_count == 1
    assert reviewed.interval_days == 1
    assert reviewed.due_at == REVIEWED_AT + timedelta(days=1)
    assert banks.calls == [(DOCUMENT_ID, BANK_IDENTITY)]
    assert progress.find_calls == [
        (DOCUMENT_ID, BANK_IDENTITY, 1),
    ]
    assert progress.saved == [reviewed]


def test_review_service_continues_existing_progress() -> None:
    existing = QuestionProgress(
        document_id=DOCUMENT_ID,
        question_bank_identity_fingerprint=BANK_IDENTITY,
        question_number=1,
        repetition_count=1,
        interval_days=1,
        ease_factor=2.5,
        due_at=REVIEWED_AT,
        last_reviewed_at=REVIEWED_AT - timedelta(days=1),
    )
    progress = RecordingProgressRepository(found=existing)
    service = ReviewService(
        banks=StaticQuestionBankCatalog(build_question_bank()),
        progress=progress,
        scheduler=ReviewScheduler(),
    )

    reviewed = service.record_review(
        DOCUMENT_ID,
        BANK_IDENTITY,
        1,
        ReviewRating.GOOD,
        reviewed_at=REVIEWED_AT,
    )

    assert reviewed.repetition_count == 2
    assert reviewed.interval_days == 6
    assert progress.saved == [reviewed]


def test_review_service_rejects_unknown_question_before_reading_progress() -> None:
    progress = RecordingProgressRepository()
    service = ReviewService(
        banks=StaticQuestionBankCatalog(build_question_bank()),
        progress=progress,
        scheduler=ReviewScheduler(),
    )

    with pytest.raises(
        StudyQuestionNotFoundError,
        match="Study question does not exist",
    ):
        service.record_review(
            DOCUMENT_ID,
            BANK_IDENTITY,
            99,
            ReviewRating.GOOD,
            reviewed_at=REVIEWED_AT,
        )

    assert progress.find_calls == []
    assert progress.saved == []


def test_list_due_includes_an_unreviewed_question() -> None:
    bank = build_question_bank()
    progress = RecordingProgressRepository()
    service = ReviewService(
        banks=StaticQuestionBankCatalog(bank),
        progress=progress,
        scheduler=ReviewScheduler(),
    )

    due = service.list_due(
        DOCUMENT_ID,
        BANK_IDENTITY,
        as_of=REVIEWED_AT,
        limit=10,
    )

    assert due == [
        DueQuestion(
            question=bank.questions[0],
            progress=None,
        )
    ]
    assert progress.find_calls == [
        (DOCUMENT_ID, BANK_IDENTITY, 1),
    ]


@pytest.mark.parametrize("limit", [0, -1])
def test_list_due_requires_positive_limit_before_loading_bank(
    limit: int,
) -> None:
    banks = StaticQuestionBankCatalog(build_question_bank())
    service = ReviewService(
        banks=banks,
        progress=RecordingProgressRepository(),
        scheduler=ReviewScheduler(),
    )

    with pytest.raises(
        ValueError,
        match="limit must be positive",
    ):
        service.list_due(
            DOCUMENT_ID,
            BANK_IDENTITY,
            as_of=REVIEWED_AT,
            limit=limit,
        )

    assert banks.calls == []


def test_list_due_requires_timezone_aware_timestamp_before_loading_bank() -> None:
    banks = StaticQuestionBankCatalog(build_question_bank())
    service = ReviewService(
        banks=banks,
        progress=RecordingProgressRepository(),
        scheduler=ReviewScheduler(),
    )

    with pytest.raises(
        ValueError,
        match="Due-query timestamp must include a timezone",
    ):
        service.list_due(
            DOCUMENT_ID,
            BANK_IDENTITY,
            as_of=datetime(2026, 8, 14, 12, 0),
            limit=10,
        )

    assert banks.calls == []


class MappedProgressRepository(RecordingProgressRepository):
    def __init__(
        self,
        progress_by_question: dict[int, QuestionProgress],
    ) -> None:
        super().__init__()
        self.progress_by_question = progress_by_question

    def find(
        self,
        document_id: UUID,
        question_bank_identity_fingerprint: str,
        question_number: int,
    ) -> QuestionProgress | None:
        self.find_calls.append(
            (
                document_id,
                question_bank_identity_fingerprint,
                question_number,
            )
        )
        return self.progress_by_question.get(question_number)


def test_list_due_prioritizes_overdue_reviews_before_new_questions() -> None:
    original = build_question_bank()
    second_question = StudyQuestion(
        number=2,
        text="Why are citations stored?",
        expected_answer="They make generated material traceable.",
        citations=original.questions[0].citations,
    )
    bank = QuestionBank(
        document_id=original.document_id,
        identity_fingerprint=original.identity_fingerprint,
        source=original.source,
        questions=(
            original.questions[0],
            second_question,
        ),
        prompt_references=original.prompt_references,
    )
    overdue = QuestionProgress(
        document_id=DOCUMENT_ID,
        question_bank_identity_fingerprint=BANK_IDENTITY,
        question_number=2,
        repetition_count=1,
        interval_days=1,
        ease_factor=2.5,
        due_at=REVIEWED_AT - timedelta(days=1),
        last_reviewed_at=REVIEWED_AT - timedelta(days=2),
    )
    service = ReviewService(
        banks=StaticQuestionBankCatalog(bank),
        progress=MappedProgressRepository({2: overdue}),
        scheduler=ReviewScheduler(),
    )

    due = service.list_due(
        DOCUMENT_ID,
        BANK_IDENTITY,
        as_of=REVIEWED_AT,
        limit=1,
    )

    assert due == [
        DueQuestion(
            question=second_question,
            progress=overdue,
        )
    ]


def test_prepare_review_calculates_progress_without_saving() -> None:
    progress = RecordingProgressRepository()
    service = ReviewService(
        banks=StaticQuestionBankCatalog(build_question_bank()),
        progress=progress,
        scheduler=ReviewScheduler(),
    )

    reviewed = service.prepare_review(
        DOCUMENT_ID,
        BANK_IDENTITY,
        1,
        ReviewRating.GOOD,
        reviewed_at=REVIEWED_AT,
    )

    assert reviewed.repetition_count == 1
    assert reviewed.interval_days == 1
    assert reviewed.due_at == REVIEWED_AT + timedelta(days=1)
    assert progress.find_calls == [
        (DOCUMENT_ID, BANK_IDENTITY, 1),
    ]
    assert progress.saved == []
