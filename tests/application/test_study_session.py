from datetime import UTC, datetime, timedelta
from uuid import UUID

import pytest

from rag_learning_assistant.application import (
    DueQuestion,
    EvaluatedStudyAnswer,
    StudyQuestionNotFoundError,
    StudySessionService,
)
from rag_learning_assistant.generation import Citation, PromptReference
from rag_learning_assistant.learning import (
    AnswerEvaluation,
    AnswerVerdict,
    QuestionBank,
    QuestionProgress,
    ReviewRating,
    StudyAttempt,
    StudyQuestion,
)

ATTEMPT_ID = UUID("dddddddd-dddd-dddd-dddd-dddddddddddd")
DOCUMENT_ID = UUID("aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa")
BANK_IDENTITY = "b" * 64
ANSWERED_AT = datetime(2026, 8, 14, 12, 0, tzinfo=UTC)


def build_bank() -> QuestionBank:
    return QuestionBank(
        document_id=DOCUMENT_ID,
        identity_fingerprint=BANK_IDENTITY,
        source="document.pdf",
        questions=(
            StudyQuestion(
                number=1,
                text="What is retrieval?",
                expected_answer=("Retrieval finds relevant source passages."),
                citations=(
                    Citation(
                        number=1,
                        source="document.pdf",
                        page_number=1,
                        chunk_index=0,
                        excerpt=("Retrieval finds relevant source passages."),
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


class StaticQuestionBankLookup:
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


class RecordingReviewer:
    def __init__(self, progress: QuestionProgress) -> None:
        self.progress = progress
        self.calls: list[tuple[UUID, str, int, ReviewRating, datetime]] = []
        self.due: list[DueQuestion] = []
        self.list_due_calls: list[tuple[UUID, str, datetime, int]] = []

    def prepare_review(
        self,
        document_id: UUID,
        question_bank_identity_fingerprint: str,
        question_number: int,
        rating: ReviewRating,
        *,
        reviewed_at: datetime,
    ) -> QuestionProgress:
        self.calls.append(
            (
                document_id,
                question_bank_identity_fingerprint,
                question_number,
                rating,
                reviewed_at,
            )
        )
        return self.progress

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
        return self.due[:limit]


class RecordingAttemptRepository:
    def __init__(self) -> None:
        self.added: list[StudyAttempt] = []

    def record(self, attempt: StudyAttempt) -> None:
        self.added.append(attempt)


def test_record_answer_schedules_and_persists_complete_attempt() -> None:
    bank = build_bank()
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
    banks = StaticQuestionBankLookup(bank)
    reviewer = RecordingReviewer(progress)
    attempts = RecordingAttemptRepository()
    service = StudySessionService(
        banks=banks,
        reviewer=reviewer,
        attempts=attempts,
        attempt_id_factory=lambda: ATTEMPT_ID,
    )

    attempt = service.record_answer(
        DOCUMENT_ID,
        BANK_IDENTITY,
        1,
        answer_text="It finds relevant passages.",
        rating=ReviewRating.GOOD,
        answered_at=ANSWERED_AT,
    )

    assert attempt == StudyAttempt(
        id=ATTEMPT_ID,
        document_id=DOCUMENT_ID,
        question_bank_identity_fingerprint=BANK_IDENTITY,
        question_number=1,
        question_text=bank.questions[0].text,
        answer_text="It finds relevant passages.",
        expected_answer=bank.questions[0].expected_answer,
        citations=bank.questions[0].citations,
        rating=ReviewRating.GOOD,
        answered_at=ANSWERED_AT,
        resulting_progress=progress,
    )
    assert banks.calls == [(DOCUMENT_ID, BANK_IDENTITY)]
    assert reviewer.calls == [
        (
            DOCUMENT_ID,
            BANK_IDENTITY,
            1,
            ReviewRating.GOOD,
            ANSWERED_AT,
        )
    ]
    assert attempts.added == [attempt]


def build_progress() -> QuestionProgress:
    return QuestionProgress(
        document_id=DOCUMENT_ID,
        question_bank_identity_fingerprint=BANK_IDENTITY,
        question_number=1,
        repetition_count=1,
        interval_days=1,
        ease_factor=2.5,
        due_at=ANSWERED_AT + timedelta(days=1),
        last_reviewed_at=ANSWERED_AT,
    )


@pytest.mark.parametrize("answer_text", ["", "   "])
def test_record_answer_rejects_blank_answer_before_side_effects(
    answer_text: str,
) -> None:
    banks = StaticQuestionBankLookup(build_bank())
    reviewer = RecordingReviewer(build_progress())
    attempts = RecordingAttemptRepository()
    generated_ids: list[UUID] = []

    def generate_id() -> UUID:
        generated_ids.append(ATTEMPT_ID)
        return ATTEMPT_ID

    service = StudySessionService(
        banks=banks,
        reviewer=reviewer,
        attempts=attempts,
        attempt_id_factory=generate_id,
    )

    with pytest.raises(
        ValueError,
        match="Study answer must not be blank",
    ):
        service.record_answer(
            DOCUMENT_ID,
            BANK_IDENTITY,
            1,
            answer_text=answer_text,
            rating=ReviewRating.GOOD,
            answered_at=ANSWERED_AT,
        )

    assert banks.calls == []
    assert reviewer.calls == []
    assert attempts.added == []
    assert generated_ids == []


def test_record_answer_rejects_naive_timestamp_before_side_effects() -> None:
    banks = StaticQuestionBankLookup(build_bank())
    reviewer = RecordingReviewer(build_progress())
    attempts = RecordingAttemptRepository()

    service = StudySessionService(
        banks=banks,
        reviewer=reviewer,
        attempts=attempts,
        attempt_id_factory=lambda: ATTEMPT_ID,
    )

    with pytest.raises(
        ValueError,
        match="Study answer timestamp must include a timezone",
    ):
        service.record_answer(
            DOCUMENT_ID,
            BANK_IDENTITY,
            1,
            answer_text="It finds relevant passages.",
            rating=ReviewRating.GOOD,
            answered_at=datetime(2026, 8, 14, 12, 0),
        )

    assert banks.calls == []
    assert reviewer.calls == []
    assert attempts.added == []


def test_record_answer_rejects_unknown_question_before_review() -> None:
    banks = StaticQuestionBankLookup(build_bank())
    reviewer = RecordingReviewer(build_progress())
    attempts = RecordingAttemptRepository()
    generated_ids: list[UUID] = []

    def generate_id() -> UUID:
        generated_ids.append(ATTEMPT_ID)
        return ATTEMPT_ID

    service = StudySessionService(
        banks=banks,
        reviewer=reviewer,
        attempts=attempts,
        attempt_id_factory=generate_id,
    )

    with pytest.raises(
        StudyQuestionNotFoundError,
        match="Study question does not exist",
    ):
        service.record_answer(
            DOCUMENT_ID,
            BANK_IDENTITY,
            99,
            answer_text="My answer.",
            rating=ReviewRating.GOOD,
            answered_at=ANSWERED_AT,
        )

    assert banks.calls == [(DOCUMENT_ID, BANK_IDENTITY)]
    assert reviewer.calls == []
    assert attempts.added == []
    assert generated_ids == []


def test_next_due_returns_highest_priority_question() -> None:
    bank = build_bank()
    reviewer = RecordingReviewer(build_progress())
    expected = DueQuestion(
        question=bank.questions[0],
        progress=None,
    )
    reviewer.due = [expected]
    service = StudySessionService(
        banks=StaticQuestionBankLookup(bank),
        reviewer=reviewer,
        attempts=RecordingAttemptRepository(),
        attempt_id_factory=lambda: ATTEMPT_ID,
    )

    due = service.next_due(
        DOCUMENT_ID,
        BANK_IDENTITY,
        as_of=ANSWERED_AT,
    )

    assert due == expected
    assert reviewer.list_due_calls == [
        (
            DOCUMENT_ID,
            BANK_IDENTITY,
            ANSWERED_AT,
            1,
        )
    ]


class RecordingAnswerEvaluator:
    def __init__(
        self,
        result: EvaluatedStudyAnswer,
    ) -> None:
        self.result = result
        self.calls: list[tuple[StudyQuestion, str]] = []

    def evaluate(
        self,
        question: StudyQuestion,
        answer_text: str,
    ) -> EvaluatedStudyAnswer:
        self.calls.append((question, answer_text))
        return self.result


def test_record_answer_uses_automatic_evaluation_rating() -> None:
    bank = build_bank()
    evaluation = AnswerEvaluation(
        verdict=AnswerVerdict.PARTIALLY_CORRECT,
        score=0.7,
        feedback="The answer omits the generation order.",
        missing_concepts=("Retrieval happens before generation.",),
    )
    evaluated = EvaluatedStudyAnswer(
        evaluation=evaluation,
        rating=ReviewRating.HARD,
        prompt_references=(),
    )
    evaluator = RecordingAnswerEvaluator(evaluated)
    reviewer = RecordingReviewer(build_progress())
    attempts = RecordingAttemptRepository()
    service = StudySessionService(
        banks=StaticQuestionBankLookup(bank),
        reviewer=reviewer,
        attempts=attempts,
        attempt_id_factory=lambda: ATTEMPT_ID,
        evaluator=evaluator,
    )

    attempt = service.record_answer(
        DOCUMENT_ID,
        BANK_IDENTITY,
        1,
        answer_text="Retrieval finds passages.",
        answered_at=ANSWERED_AT,
    )

    assert evaluator.calls == [
        (
            bank.questions[0],
            "Retrieval finds passages.",
        )
    ]
    assert reviewer.calls == [
        (
            DOCUMENT_ID,
            BANK_IDENTITY,
            1,
            ReviewRating.HARD,
            ANSWERED_AT,
        )
    ]
    assert attempt.rating is ReviewRating.HARD
    assert attempt.evaluation == evaluation
    assert attempts.added == [attempt]


class FailingAnswerEvaluator:
    def __init__(self) -> None:
        self.calls: list[tuple[StudyQuestion, str]] = []

    def evaluate(
        self,
        question: StudyQuestion,
        answer_text: str,
    ) -> EvaluatedStudyAnswer:
        self.calls.append((question, answer_text))
        raise ValueError("Model evaluation must be valid JSON")


def test_failed_automatic_evaluation_preserves_schedule_and_history() -> None:
    bank = build_bank()
    evaluator = FailingAnswerEvaluator()
    reviewer = RecordingReviewer(build_progress())
    attempts = RecordingAttemptRepository()
    generated_ids: list[UUID] = []

    def generate_id() -> UUID:
        generated_ids.append(ATTEMPT_ID)
        return ATTEMPT_ID

    service = StudySessionService(
        banks=StaticQuestionBankLookup(bank),
        reviewer=reviewer,
        attempts=attempts,
        attempt_id_factory=generate_id,
        evaluator=evaluator,
    )

    with pytest.raises(
        ValueError,
        match="Model evaluation must be valid JSON",
    ):
        service.record_answer(
            DOCUMENT_ID,
            BANK_IDENTITY,
            1,
            answer_text="Retrieval finds passages.",
            answered_at=ANSWERED_AT,
        )

    assert evaluator.calls == [
        (
            bank.questions[0],
            "Retrieval finds passages.",
        )
    ]
    assert reviewer.calls == []
    assert attempts.added == []
    assert generated_ids == []
