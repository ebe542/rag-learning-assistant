from datetime import UTC, datetime, timedelta, timezone
from pathlib import Path
from uuid import UUID

from fastapi.testclient import TestClient

from rag_learning_assistant.application import DueQuestion, LearningPackageNotFoundError
from rag_learning_assistant.generation import Citation, PersistedDocumentSummary, PromptReference
from rag_learning_assistant.interfaces.web import create_app
from rag_learning_assistant.interfaces.web.application import format_local_datetime
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


class StubPackageCatalog:
    def __init__(self, packages: list[LearningPackage]) -> None:
        self.packages = packages

    def list_packages(self) -> list[LearningPackage]:
        return self.packages

    def get_package(self, name: str) -> LearningPackage:
        for package in self.packages:
            if package.name.casefold() == name.casefold():
                return package
        raise LearningPackageNotFoundError(f"Learning package not found: {name}")


class StubSummaryCatalog:
    def __init__(self, summary: PersistedDocumentSummary | None = None) -> None:
        self.summary = summary

    def get_document_summary(
        self,
        document_id: UUID,
        identity_fingerprint: str,
    ) -> PersistedDocumentSummary:
        assert self.summary is not None
        assert self.summary.document_id == document_id
        assert self.summary.identity_fingerprint == identity_fingerprint
        return self.summary


class StubQuestionCatalog:
    def __init__(self, bank: QuestionBank | None = None) -> None:
        self.bank = bank

    def get_document_bank(
        self,
        document_id: UUID,
        identity_fingerprint: str,
    ) -> QuestionBank:
        assert self.bank is not None
        assert self.bank.document_id == document_id
        assert self.bank.identity_fingerprint == identity_fingerprint
        return self.bank


class StubPackageStudy:
    def __init__(
        self,
        due: DueQuestion | None = None,
        attempt: StudyAttempt | None = None,
    ) -> None:
        self.due = due
        self.attempt = attempt

    def next_due(self, package_name: str, *, as_of: datetime) -> DueQuestion | None:
        return self.due

    def record_answer(
        self,
        package_name: str,
        question_number: int,
        *,
        answer_text: str,
        answered_at: datetime,
    ) -> StudyAttempt:
        if self.attempt is None:
            raise AssertionError("No study attempt configured")
        return self.attempt


def build_client(
    packages: list[LearningPackage] | None = None,
    study: StubPackageStudy | None = None,
) -> TestClient:
    return TestClient(
        create_app(
            StubPackageCatalog(packages or []),
            StubSummaryCatalog(),
            StubQuestionCatalog(),
            study or StubPackageStudy(),
            library_directory=Path("personal-library"),
        )
    )


def ready_package() -> LearningPackage:
    return LearningPackage(
        id=UUID("aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa"),
        name="Python Basics",
        document_id=UUID("bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb"),
        status=LearningPackageStatus.READY,
        summary_identity_fingerprint="a" * 64,
        question_bank_identity_fingerprint="b" * 64,
    )


def prompt_reference() -> PromptReference:
    return PromptReference(name="test.prompt", version=1, fingerprint="c" * 64)


def citation() -> Citation:
    return Citation(
        number=1,
        source="python.pdf",
        page_number=3,
        chunk_index=2,
        excerpt="Functions group reusable instructions.",
    )


def test_home_page_introduces_local_learning_workspace() -> None:
    response = build_client().get("/")

    assert response.status_code == 200
    assert "RAG Learning Assistant" in response.text
    assert "Learning packages" in response.text
    assert "No learning packages yet" in response.text
    assert "personal-library" in response.text


def test_home_page_lists_packages_with_their_preparation_status() -> None:
    response = build_client([ready_package()]).get("/")

    assert response.status_code == 200
    assert "Python Basics" in response.text
    assert "Ready" in response.text
    assert "No learning packages yet" not in response.text
    assert "/package?name=Python%20Basics" in response.text


def test_package_detail_shows_summary_and_question_count() -> None:
    package = ready_package()
    summary = PersistedDocumentSummary(
        document_id=package.document_id,
        identity_fingerprint=package.summary_identity_fingerprint or "",
        source="python.pdf",
        text="Functions organize code into reusable units.",
        citations=(citation(),),
        prompt_references=(prompt_reference(),),
    )
    bank = QuestionBank(
        document_id=package.document_id,
        identity_fingerprint=package.question_bank_identity_fingerprint or "",
        source="python.pdf",
        questions=(
            StudyQuestion(
                number=1,
                text="What is a function?",
                expected_answer="A reusable group of instructions.",
                citations=(citation(),),
            ),
        ),
        prompt_references=(prompt_reference(),),
    )
    client = TestClient(
        create_app(
            StubPackageCatalog([package]),
            StubSummaryCatalog(summary),
            StubQuestionCatalog(bank),
            StubPackageStudy(),
            library_directory=Path("personal-library"),
        )
    )

    response = client.get("/package", params={"name": "Python Basics"})

    assert response.status_code == 200
    assert "Functions organize code into reusable units." in response.text
    assert "Questions" in response.text
    assert ">1<" in response.text
    assert str(package.document_id) in response.text


def test_unknown_package_returns_a_helpful_not_found_page() -> None:
    response = build_client().get("/package", params={"name": "Missing"})

    assert response.status_code == 404
    assert "Learning package not found" in response.text
    assert "Missing" in response.text


def test_study_page_shows_the_next_due_question() -> None:
    question = StudyQuestion(
        number=1,
        text="What is a function?",
        expected_answer="A reusable group of instructions.",
        citations=(citation(),),
    )
    response = build_client(
        [ready_package()],
        StubPackageStudy(DueQuestion(question=question, progress=None)),
    ).get("/study", params={"package": "Python Basics"})

    assert response.status_code == 200
    assert "What is a function?" in response.text
    assert "Your answer" in response.text
    assert "expected_answer" not in response.text
    assert "data-study-form" in response.text
    assert "data-submit-button" in response.text
    assert "data-submission-status" in response.text


def test_study_submission_rejects_a_cross_origin_request() -> None:
    question = StudyQuestion(
        number=1,
        text="What is a function?",
        expected_answer="A reusable group of instructions.",
        citations=(citation(),),
    )
    response = build_client(
        [ready_package()],
        StubPackageStudy(DueQuestion(question=question, progress=None)),
    ).post(
        "/study",
        data={"package": "Python Basics", "question_number": "1", "answer": "Reusable code"},
        headers={"origin": "https://attacker.example"},
    )

    assert response.status_code == 403


def test_study_submission_renders_grounded_feedback() -> None:
    question = StudyQuestion(
        number=1,
        text="What is a function?",
        expected_answer="A reusable group of instructions.",
        citations=(citation(),),
    )
    answered_at = datetime(2026, 8, 28, 12, 0, tzinfo=UTC)
    progress = QuestionProgress(
        document_id=ready_package().document_id,
        question_bank_identity_fingerprint="b" * 64,
        question_number=1,
        repetition_count=1,
        interval_days=1,
        ease_factor=2.5,
        due_at=answered_at + timedelta(days=1),
        last_reviewed_at=answered_at,
    )
    attempt = StudyAttempt(
        id=UUID("dddddddd-dddd-dddd-dddd-dddddddddddd"),
        document_id=progress.document_id,
        question_bank_identity_fingerprint=progress.question_bank_identity_fingerprint,
        question_number=1,
        question_text=question.text,
        answer_text="Reusable code.",
        expected_answer=question.expected_answer,
        citations=question.citations,
        rating=ReviewRating.GOOD,
        answered_at=answered_at,
        resulting_progress=progress,
        evaluation=AnswerEvaluation(
            verdict=AnswerVerdict.CORRECT,
            score=1.0,
            feedback="The answer captures the central idea.",
            missing_concepts=(),
        ),
    )
    response = build_client(
        [ready_package()],
        StubPackageStudy(DueQuestion(question=question, progress=None), attempt),
    ).post(
        "/study",
        data={"package": "Python Basics", "question_number": "1", "answer": "Reusable code"},
        headers={"origin": "http://testserver"},
    )

    assert response.status_code == 200
    assert "Study feedback" in response.text
    assert "The answer captures the central idea." in response.text
    assert format_local_datetime(progress.due_at) in response.text
    assert progress.due_at.isoformat() not in response.text


def test_datetime_format_shows_the_converted_local_time() -> None:
    value = datetime(2026, 8, 28, 13, 1, tzinfo=UTC)

    assert (
        format_local_datetime(
            value,
            timezone=timezone(timedelta(hours=2)),
        )
        == "28.08.2026, 15:01 (local time)"
    )


def test_static_styles_are_served_locally() -> None:
    response = build_client().get("/static/app.css")

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/css")
    assert ".panel" in response.text


def test_study_script_is_served_locally_and_prevents_duplicate_submissions() -> None:
    response = build_client().get("/static/study.js")

    assert response.status_code == 200
    assert "button.disabled = true" in response.text
    assert 'form.setAttribute("aria-busy", "true")' in response.text
    assert "status.hidden = false" in response.text
