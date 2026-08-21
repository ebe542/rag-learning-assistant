from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from uuid import UUID

import pytest

from rag_learning_assistant.application import (
    LearningPackageNotFoundError,
    LearningPackageNotReadyError,
    LearningProgressReport,
    LearningProgressService,
)
from rag_learning_assistant.generation import Citation, PromptReference
from rag_learning_assistant.learning import (
    AnswerEvaluation,
    AnswerVerdict,
    LearningPackage,
    LearningPackageStatus,
    QuestionBank,
    QuestionProgress,
    ReviewRating,
    StudyAttempt,
    StudyQuestion,
)

LAST_STUDIED_AT = datetime(
    2026,
    8,
    20,
    12,
    0,
    tzinfo=UTC,
)
NEXT_DUE_AT = datetime(
    2026,
    8,
    21,
    12,
    0,
    tzinfo=UTC,
)
DOCUMENT_ID = UUID("aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa")
BANK_IDENTITY = "b" * 64


def build_report(
    *,
    package_name: str = "RAG Learning Assistant",
    total_question_count: int = 10,
    answered_question_count: int = 4,
    due_question_count: int = 3,
    attempt_count: int = 6,
    incorrect_attempt_count: int = 2,
    partially_correct_attempt_count: int = 1,
    correct_attempt_count: int = 3,
    unclassified_attempt_count: int = 0,
) -> LearningProgressReport:
    return LearningProgressReport(
        package_name=package_name,
        total_question_count=total_question_count,
        answered_question_count=answered_question_count,
        due_question_count=due_question_count,
        attempt_count=attempt_count,
        incorrect_attempt_count=incorrect_attempt_count,
        partially_correct_attempt_count=partially_correct_attempt_count,
        correct_attempt_count=correct_attempt_count,
        difficult_concepts=(
            ("document identity", 2),
            ("citation relationships", 1),
        ),
        last_studied_at=LAST_STUDIED_AT,
        next_due_at=NEXT_DUE_AT,
        unclassified_attempt_count=unclassified_attempt_count,
    )


def test_learning_progress_report_exposes_package_metrics() -> None:
    report = build_report()

    assert report.package_name == "RAG Learning Assistant"
    assert report.total_question_count == 10
    assert report.answered_question_count == 4
    assert report.due_question_count == 3
    assert report.attempt_count == 6
    assert report.difficult_concepts == (
        ("document identity", 2),
        ("citation relationships", 1),
    )
    assert report.last_studied_at == LAST_STUDIED_AT
    assert report.next_due_at == NEXT_DUE_AT


@pytest.mark.parametrize("package_name", ["", "   "])
def test_learning_progress_report_rejects_blank_package_name(
    package_name: str,
) -> None:
    with pytest.raises(
        ValueError,
        match="Progress package name must not be blank",
    ):
        build_report(package_name=package_name)


@pytest.mark.parametrize(
    "build_invalid_report",
    [
        lambda: build_report(total_question_count=-1),
        lambda: build_report(answered_question_count=-1),
        lambda: build_report(due_question_count=-1),
        lambda: build_report(attempt_count=-1),
        lambda: build_report(incorrect_attempt_count=-1),
        lambda: build_report(partially_correct_attempt_count=-1),
        lambda: build_report(correct_attempt_count=-1),
        lambda: build_report(unclassified_attempt_count=-1),
    ],
)
def test_learning_progress_report_rejects_negative_counts(
    build_invalid_report: Callable[
        [],
        LearningProgressReport,
    ],
) -> None:
    with pytest.raises(
        ValueError,
        match="Progress counts must not be negative",
    ):
        build_invalid_report()


def test_learning_progress_report_requires_complete_attempt_breakdown() -> None:
    with pytest.raises(
        ValueError,
        match="Progress attempt counts must equal total attempts",
    ):
        build_report(
            attempt_count=7,
            incorrect_attempt_count=2,
            partially_correct_attempt_count=1,
            correct_attempt_count=3,
        )


def test_learning_progress_report_calculates_rates() -> None:
    report = build_report()

    assert report.answered_rate == pytest.approx(0.4)
    assert report.correct_attempt_rate == pytest.approx(0.5)


def test_learning_progress_report_uses_zero_rates_without_data() -> None:
    report = build_report(
        total_question_count=0,
        answered_question_count=0,
        due_question_count=0,
        attempt_count=0,
        incorrect_attempt_count=0,
        partially_correct_attempt_count=0,
        correct_attempt_count=0,
    )

    assert report.answered_rate == 0.0
    assert report.correct_attempt_rate == 0.0


class MissingPackageLookup:
    def find_by_name(
        self,
        name: str,
    ) -> LearningPackage | None:
        return None


class UnusedQuestionBankLookup:
    def get_document_bank(
        self,
        document_id: UUID,
        identity_fingerprint: str,
    ) -> QuestionBank:
        raise AssertionError("Question bank must not be loaded")


class UnusedProgressReader:
    def find(
        self,
        document_id: UUID,
        question_bank_identity_fingerprint: str,
        question_number: int,
    ) -> QuestionProgress | None:
        raise AssertionError("Progress must not be loaded")


class UnusedAttemptReader:
    def list_question(
        self,
        document_id: UUID,
        question_bank_identity_fingerprint: str,
        question_number: int,
    ) -> list[StudyAttempt]:
        raise AssertionError("Attempts must not be loaded")


def test_learning_progress_rejects_unknown_package_before_loading_data() -> None:
    service = LearningProgressService(
        packages=MissingPackageLookup(),
        banks=UnusedQuestionBankLookup(),
        progress=UnusedProgressReader(),
        attempts=UnusedAttemptReader(),
    )

    with pytest.raises(
        LearningPackageNotFoundError,
        match="Learning package does not exist: Unknown package",
    ):
        service.report(
            "Unknown package",
            as_of=LAST_STUDIED_AT,
        )


class StaticPackageLookup:
    def __init__(self, package: LearningPackage) -> None:
        self.package = package

    def find_by_name(
        self,
        name: str,
    ) -> LearningPackage | None:
        if self.package.name.casefold() == name.casefold():
            return self.package

        return None


class StaticQuestionBankLookup:
    def __init__(self, bank: QuestionBank) -> None:
        self.bank = bank

    def get_document_bank(
        self,
        document_id: UUID,
        identity_fingerprint: str,
    ) -> QuestionBank:
        assert document_id == self.bank.document_id
        assert identity_fingerprint == self.bank.identity_fingerprint
        return self.bank


class EmptyProgressReader:
    def find(
        self,
        document_id: UUID,
        question_bank_identity_fingerprint: str,
        question_number: int,
    ) -> QuestionProgress | None:
        return None


class EmptyAttemptReader:
    def list_question(
        self,
        document_id: UUID,
        question_bank_identity_fingerprint: str,
        question_number: int,
    ) -> list[StudyAttempt]:
        return []


def build_ready_package() -> LearningPackage:
    return LearningPackage(
        id=UUID("cccccccc-cccc-cccc-cccc-cccccccccccc"),
        name="RAG Learning Assistant",
        document_id=DOCUMENT_ID,
        status=LearningPackageStatus.READY,
        summary_identity_fingerprint="d" * 64,
        question_bank_identity_fingerprint=BANK_IDENTITY,
    )


def build_question_bank() -> QuestionBank:
    citation = Citation(
        number=1,
        source="document.pdf",
        page_number=1,
        chunk_index=0,
        excerpt="Retrieval finds relevant passages.",
    )

    return QuestionBank(
        document_id=DOCUMENT_ID,
        identity_fingerprint=BANK_IDENTITY,
        source="document.pdf",
        questions=(
            StudyQuestion(
                number=1,
                text="What is retrieval?",
                expected_answer=("Retrieval finds relevant source passages."),
                citations=(citation,),
            ),
        ),
        prompt_references=(
            PromptReference(
                name="question-bank",
                version=1,
                fingerprint="e" * 64,
            ),
        ),
    )


def test_learning_progress_reports_new_questions_as_due() -> None:
    package = build_ready_package()
    service = LearningProgressService(
        packages=StaticPackageLookup(package),
        banks=StaticQuestionBankLookup(build_question_bank()),
        progress=EmptyProgressReader(),
        attempts=EmptyAttemptReader(),
    )

    report = service.report(
        "rag learning assistant",
        as_of=LAST_STUDIED_AT,
    )

    assert report == LearningProgressReport(
        package_name="RAG Learning Assistant",
        total_question_count=1,
        answered_question_count=0,
        due_question_count=1,
        attempt_count=0,
        incorrect_attempt_count=0,
        partially_correct_attempt_count=0,
        correct_attempt_count=0,
        difficult_concepts=(),
        last_studied_at=None,
        next_due_at=LAST_STUDIED_AT,
        unclassified_attempt_count=0,
    )


class StaticProgressReader:
    def __init__(
        self,
        progress: QuestionProgress,
    ) -> None:
        self.progress = progress

    def find(
        self,
        document_id: UUID,
        question_bank_identity_fingerprint: str,
        question_number: int,
    ) -> QuestionProgress | None:
        return self.progress


class StaticAttemptReader:
    def __init__(
        self,
        attempts: list[StudyAttempt],
    ) -> None:
        self.attempts = attempts

    def list_question(
        self,
        document_id: UUID,
        question_bank_identity_fingerprint: str,
        question_number: int,
    ) -> list[StudyAttempt]:
        return self.attempts


def test_learning_progress_aggregates_attempts_and_schedule() -> None:
    first_answered_at = LAST_STUDIED_AT - timedelta(days=1)
    first_progress = QuestionProgress(
        document_id=DOCUMENT_ID,
        question_bank_identity_fingerprint=BANK_IDENTITY,
        question_number=1,
        repetition_count=1,
        interval_days=1,
        ease_factor=2.5,
        due_at=LAST_STUDIED_AT,
        last_reviewed_at=first_answered_at,
    )
    progress = QuestionProgress(
        document_id=DOCUMENT_ID,
        question_bank_identity_fingerprint=BANK_IDENTITY,
        question_number=1,
        repetition_count=2,
        interval_days=1,
        ease_factor=2.5,
        due_at=NEXT_DUE_AT,
        last_reviewed_at=LAST_STUDIED_AT,
    )
    question = build_question_bank().questions[0]

    first_attempt = StudyAttempt(
        id=UUID("11111111-1111-1111-1111-111111111111"),
        document_id=DOCUMENT_ID,
        question_bank_identity_fingerprint=BANK_IDENTITY,
        question_number=1,
        question_text=question.text,
        answer_text="It searches documents.",
        expected_answer=question.expected_answer,
        citations=question.citations,
        rating=ReviewRating.HARD,
        answered_at=first_answered_at,
        resulting_progress=first_progress,
        evaluation=AnswerEvaluation(
            verdict=AnswerVerdict.PARTIALLY_CORRECT,
            score=0.6,
            feedback="The answer omits relevance.",
            missing_concepts=("relevant passages",),
        ),
    )
    second_attempt = StudyAttempt(
        id=UUID("22222222-2222-2222-2222-222222222222"),
        document_id=DOCUMENT_ID,
        question_bank_identity_fingerprint=BANK_IDENTITY,
        question_number=1,
        question_text=question.text,
        answer_text="It finds relevant passages.",
        expected_answer=question.expected_answer,
        citations=question.citations,
        rating=ReviewRating.GOOD,
        answered_at=LAST_STUDIED_AT,
        resulting_progress=progress,
        evaluation=AnswerEvaluation(
            verdict=AnswerVerdict.CORRECT,
            score=1.0,
            feedback="Correct.",
            missing_concepts=(),
        ),
    )
    service = LearningProgressService(
        packages=StaticPackageLookup(build_ready_package()),
        banks=StaticQuestionBankLookup(build_question_bank()),
        progress=StaticProgressReader(progress),
        attempts=StaticAttemptReader([first_attempt, second_attempt]),
    )

    report = service.report(
        "RAG Learning Assistant",
        as_of=LAST_STUDIED_AT,
    )

    assert report.total_question_count == 1
    assert report.answered_question_count == 1
    assert report.due_question_count == 0
    assert report.attempt_count == 2
    assert report.incorrect_attempt_count == 0
    assert report.partially_correct_attempt_count == 1
    assert report.correct_attempt_count == 1
    assert report.unclassified_attempt_count == 0
    assert report.difficult_concepts == (("relevant passages", 1),)
    assert report.last_studied_at == LAST_STUDIED_AT
    assert report.next_due_at == NEXT_DUE_AT


class RecordingPackageLookup:
    def __init__(self) -> None:
        self.names: list[str] = []

    def find_by_name(
        self,
        name: str,
    ) -> LearningPackage | None:
        self.names.append(name)
        return None


@pytest.mark.parametrize("package_name", ["", "   "])
def test_learning_progress_rejects_blank_package_name_before_lookup(
    package_name: str,
) -> None:
    packages = RecordingPackageLookup()
    service = LearningProgressService(
        packages=packages,
        banks=UnusedQuestionBankLookup(),
        progress=UnusedProgressReader(),
        attempts=UnusedAttemptReader(),
    )

    with pytest.raises(
        ValueError,
        match="Progress package name must not be blank",
    ):
        service.report(
            package_name,
            as_of=LAST_STUDIED_AT,
        )

    assert packages.names == []


def test_learning_progress_rejects_naive_timestamp_before_lookup() -> None:
    packages = RecordingPackageLookup()
    service = LearningProgressService(
        packages=packages,
        banks=UnusedQuestionBankLookup(),
        progress=UnusedProgressReader(),
        attempts=UnusedAttemptReader(),
    )

    with pytest.raises(
        ValueError,
        match="Progress timestamp must include a timezone",
    ):
        service.report(
            "RAG Learning Assistant",
            as_of=datetime(2026, 8, 20, 12, 0),
        )

    assert packages.names == []


def test_learning_progress_rejects_package_without_question_bank() -> None:
    package = LearningPackage(
        id=UUID("cccccccc-cccc-cccc-cccc-cccccccccccc"),
        name="RAG Learning Assistant",
        document_id=DOCUMENT_ID,
        status=LearningPackageStatus.INDEXED,
    )
    service = LearningProgressService(
        packages=StaticPackageLookup(package),
        banks=UnusedQuestionBankLookup(),
        progress=UnusedProgressReader(),
        attempts=UnusedAttemptReader(),
    )

    with pytest.raises(
        LearningPackageNotReadyError,
        match="Learning package is not ready: RAG Learning Assistant",
    ):
        service.report(
            "RAG Learning Assistant",
            as_of=LAST_STUDIED_AT,
        )
