from datetime import UTC, datetime, timedelta
from uuid import UUID

import pytest

from rag_learning_assistant.application import (
    DueQuestion,
    LearningPackageNotFoundError,
    LearningPackageNotReadyError,
    LearningPackageStudyService,
)
from rag_learning_assistant.generation import Citation
from rag_learning_assistant.learning import (
    LearningPackage,
    LearningPackageStatus,
    QuestionProgress,
    ReviewRating,
    StudyAttempt,
)

AS_OF = datetime(2026, 8, 20, 12, 0, tzinfo=UTC)


def build_attempt() -> StudyAttempt:
    progress = QuestionProgress(
        document_id=UUID("bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb"),
        question_bank_identity_fingerprint="d" * 64,
        question_number=1,
        repetition_count=1,
        interval_days=1,
        ease_factor=2.5,
        due_at=AS_OF + timedelta(days=1),
        last_reviewed_at=AS_OF,
    )

    return StudyAttempt(
        id=UUID("eeeeeeee-eeee-eeee-eeee-eeeeeeeeeeee"),
        document_id=progress.document_id,
        question_bank_identity_fingerprint=(progress.question_bank_identity_fingerprint),
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
        answered_at=AS_OF,
        resulting_progress=progress,
    )


class StaticLearningPackageLookup:
    def __init__(
        self,
        package: LearningPackage,
    ) -> None:
        self.package = package
        self.names: list[str] = []

    def find_by_name(
        self,
        name: str,
    ) -> LearningPackage | None:
        self.names.append(name)

        if self.package.name.casefold() == name.casefold():
            return self.package

        return None


class RecordingStudySession:
    def __init__(self, attempt: StudyAttempt | None = None) -> None:
        self.calls: list[tuple[UUID, str, datetime]] = []
        self.record_calls: list[
            tuple[
                UUID,
                str,
                int,
                str,
                datetime,
                ReviewRating | None,
            ]
        ] = []
        self.attempt = attempt

    def next_due(
        self,
        document_id: UUID,
        question_bank_identity_fingerprint: str,
        *,
        as_of: datetime,
    ) -> DueQuestion | None:
        self.calls.append(
            (
                document_id,
                question_bank_identity_fingerprint,
                as_of,
            )
        )
        return None

    def record_answer(
        self,
        document_id: UUID,
        question_bank_identity_fingerprint: str,
        question_number: int,
        *,
        answer_text: str,
        answered_at: datetime,
        rating: ReviewRating | None = None,
    ) -> StudyAttempt:
        self.record_calls.append(
            (
                document_id,
                question_bank_identity_fingerprint,
                question_number,
                answer_text,
                answered_at,
                rating,
            )
        )

        if self.attempt is None:
            raise AssertionError("No study attempt configured")

        return self.attempt


def test_next_due_resolves_ready_package_by_name() -> None:
    package = LearningPackage(
        id=UUID("aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa"),
        name="RAG Learning Assistant",
        document_id=UUID("bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb"),
        status=LearningPackageStatus.READY,
        summary_identity_fingerprint="c" * 64,
        question_bank_identity_fingerprint="d" * 64,
    )
    packages = StaticLearningPackageLookup(package)
    sessions = RecordingStudySession()
    service = LearningPackageStudyService(
        packages=packages,
        sessions=sessions,
    )

    result = service.next_due(
        "rag learning assistant",
        as_of=AS_OF,
    )

    assert result is None
    assert packages.names == ["rag learning assistant"]
    assert sessions.calls == [
        (
            package.document_id,
            "d" * 64,
            AS_OF,
        )
    ]


def test_record_answer_resolves_ready_package_by_name() -> None:
    package = LearningPackage(
        id=UUID("aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa"),
        name="RAG Learning Assistant",
        document_id=UUID("bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb"),
        status=LearningPackageStatus.READY,
        summary_identity_fingerprint="c" * 64,
        question_bank_identity_fingerprint="d" * 64,
    )
    attempt = build_attempt()
    packages = StaticLearningPackageLookup(package)
    sessions = RecordingStudySession(attempt)
    service = LearningPackageStudyService(
        packages=packages,
        sessions=sessions,
    )

    result = service.record_answer(
        "RAG Learning Assistant",
        1,
        answer_text="It finds relevant passages.",
        answered_at=AS_OF,
    )

    assert result == attempt
    assert sessions.record_calls == [
        (
            package.document_id,
            "d" * 64,
            1,
            "It finds relevant passages.",
            AS_OF,
            None,
        )
    ]


def test_next_due_rejects_unknown_package() -> None:
    package = LearningPackage(
        id=UUID("aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa"),
        name="RAG Learning Assistant",
        document_id=UUID("bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb"),
        status=LearningPackageStatus.READY,
        summary_identity_fingerprint="c" * 64,
        question_bank_identity_fingerprint="d" * 64,
    )
    packages = StaticLearningPackageLookup(package)
    sessions = RecordingStudySession()
    service = LearningPackageStudyService(
        packages=packages,
        sessions=sessions,
    )

    with pytest.raises(
        LearningPackageNotFoundError,
        match="Learning package does not exist: Unknown package",
    ):
        service.next_due(
            "Unknown package",
            as_of=AS_OF,
        )

    assert sessions.calls == []


def test_next_due_rejects_package_that_is_not_ready() -> None:
    package = LearningPackage(
        id=UUID("aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa"),
        name="RAG Learning Assistant",
        document_id=UUID("bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb"),
        status=LearningPackageStatus.INDEXED,
        summary_identity_fingerprint=None,
        question_bank_identity_fingerprint=None,
    )
    packages = StaticLearningPackageLookup(package)
    sessions = RecordingStudySession()
    service = LearningPackageStudyService(
        packages=packages,
        sessions=sessions,
    )

    with pytest.raises(
        LearningPackageNotReadyError,
        match="Learning package is not ready: RAG Learning Assistant",
    ):
        service.next_due(
            "RAG Learning Assistant",
            as_of=AS_OF,
        )

    assert sessions.calls == []


@pytest.mark.parametrize("package_name", ["", "   "])
def test_next_due_rejects_blank_package_name_before_lookup(
    package_name: str,
) -> None:
    package = LearningPackage(
        id=UUID("aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa"),
        name="RAG Learning Assistant",
        document_id=UUID("bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb"),
        status=LearningPackageStatus.READY,
        summary_identity_fingerprint="c" * 64,
        question_bank_identity_fingerprint="d" * 64,
    )
    packages = StaticLearningPackageLookup(package)
    sessions = RecordingStudySession()
    service = LearningPackageStudyService(
        packages=packages,
        sessions=sessions,
    )

    with pytest.raises(
        ValueError,
        match="Learning package name must not be blank",
    ):
        service.next_due(
            package_name,
            as_of=AS_OF,
        )

    assert packages.names == []
    assert sessions.calls == []
