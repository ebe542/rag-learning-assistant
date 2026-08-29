from datetime import UTC, datetime, timedelta, timezone
from pathlib import Path
from uuid import UUID

from fastapi.testclient import TestClient

from rag_learning_assistant.application import (
    DueQuestion,
    LearningPackageNotFoundError,
    LearningProgressReport,
)
from rag_learning_assistant.generation import Citation, PersistedDocumentSummary, PromptReference
from rag_learning_assistant.interfaces.web import application as web_application
from rag_learning_assistant.interfaces.web import create_app
from rag_learning_assistant.interfaces.web.application import format_local_datetime
from rag_learning_assistant.interfaces.web.libraries import LibraryListItem
from rag_learning_assistant.learning import (
    AnswerEvaluation,
    AnswerVerdict,
    LearningPackage,
    LearningPackageStatus,
    PackagePreparation,
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


class StubProgressReporting:
    def __init__(self, report: LearningProgressReport | None = None) -> None:
        self.progress_report = report or LearningProgressReport(
            package_name="Python Basics",
            total_question_count=0,
            answered_question_count=0,
            due_question_count=0,
            attempt_count=0,
            incorrect_attempt_count=0,
            partially_correct_attempt_count=0,
            correct_attempt_count=0,
            difficult_concepts=(),
            last_studied_at=None,
            next_due_at=None,
            unclassified_attempt_count=0,
        )

    def report(self, package_name: str, *, as_of: datetime) -> LearningProgressReport:
        assert package_name == self.progress_report.package_name
        assert as_of.tzinfo is UTC
        return self.progress_report


class StubLibraryManagement:
    def __init__(self) -> None:
        self._current_directory = Path("personal-library")
        self.items = [
            LibraryListItem(
                UUID("11111111-1111-1111-1111-111111111111"),
                "Personal Library",
                self._current_directory,
                False,
            )
        ]
        self.preparations: list[PackagePreparation] = []
        self.uploaded_content = b""

    @property
    def current_directory(self) -> Path | None:
        return self._current_directory

    @property
    def current_library(self) -> LibraryListItem | None:
        return next(
            (item for item in self.items if item.directory == self._current_directory),
            None,
        )

    def list_libraries(self) -> tuple[LibraryListItem, ...]:
        return tuple(self.items)

    def create_library(self, name: str) -> LibraryListItem:
        if not name.strip():
            raise ValueError("Library name must not be blank")
        created = LibraryListItem(
            UUID("22222222-2222-2222-2222-222222222222"),
            name,
            Path("22222222-2222-2222-2222-222222222222"),
            False,
        )
        self.items.append(created)
        return created

    def select_library(self, library_id: UUID) -> LibraryListItem:
        selected = next((item for item in self.items if item.id == library_id), None)
        if selected is None:
            raise LookupError(f"Library not found: {library_id}")
        self._current_directory = selected.directory
        return selected

    def rename_library(self, library_id: UUID, name: str) -> LibraryListItem:
        selected = next((item for item in self.items if item.id == library_id), None)
        if selected is None:
            raise LookupError(f"Library not found: {library_id}")
        renamed = LibraryListItem(selected.id, name, selected.directory, selected.has_content)
        self.items = [renamed if item.id == library_id else item for item in self.items]
        return renamed

    def delete_library(
        self,
        library_id: UUID,
        *,
        confirmation: str,
        delete_contents: bool,
    ) -> None:
        selected = next((item for item in self.items if item.id == library_id), None)
        if selected is None:
            raise LookupError(f"Library not found: {library_id}")
        if confirmation != selected.name:
            raise ValueError("Library name confirmation does not match")
        self.items = [item for item in self.items if item.id != library_id]
        if selected.directory == self._current_directory:
            self._current_directory = self.items[0].directory if self.items else None

    def list_package_preparations(self) -> list[PackagePreparation]:
        return self.preparations

    def store_package_upload(
        self,
        *,
        name: str,
        source_filename: str,
        question_count: int,
        size_bytes: int,
        source,
    ) -> PackagePreparation:
        preparation_id = UUID("cccccccc-cccc-cccc-cccc-cccccccccccc")
        preparation = PackagePreparation(
            id=preparation_id,
            name=name,
            source_filename=source_filename,
            stored_filename=f"{preparation_id}.pdf",
            question_count=question_count,
            size_bytes=size_bytes,
        )
        self.uploaded_content = source.read()
        self.preparations.append(preparation)
        return preparation


def build_client(
    packages: list[LearningPackage] | None = None,
    study: StubPackageStudy | None = None,
    progress: StubProgressReporting | None = None,
    libraries: StubLibraryManagement | None = None,
) -> TestClient:
    library_management = libraries or StubLibraryManagement()
    return TestClient(
        create_app(
            StubPackageCatalog(packages or []),
            StubSummaryCatalog(),
            StubQuestionCatalog(),
            study or StubPackageStudy(),
            progress or StubProgressReporting(),
            libraries=library_management,
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
    assert "Your libraries" in response.text
    assert "Available libraries" in response.text
    assert "Personal Library" in response.text
    assert "Manage libraries" in response.text
    assert 'aria-label="Main navigation"' in response.text
    assert "Personal Library · Packages" in response.text


def test_library_page_lists_packages_with_their_preparation_status() -> None:
    response = build_client([ready_package()]).get("/library")

    assert response.status_code == 200
    assert "Python Basics" in response.text
    assert "Ready" in response.text
    assert "No learning packages yet" not in response.text
    assert "/package?name=Python%20Basics" in response.text
    assert "Add package" in response.text
    assert '<h2 class="heading-with-count" id="packages-heading">' in response.text
    assert '<span class="package-count">1</span>' in response.text


def test_package_create_page_shows_upload_constraints() -> None:
    response = build_client().get("/package/new")

    assert response.status_code == 200
    assert "Add learning package" in response.text
    assert 'enctype="multipart/form-data"' in response.text
    assert "Maximum PDF size: 25 MiB" in response.text


def test_valid_pdf_upload_is_stored_as_pending_without_model_processing() -> None:
    libraries = StubLibraryManagement()
    client = build_client(libraries=libraries)

    response = client.post(
        "/package/new",
        data={"name": "Python Course", "question_count": "7"},
        files={"pdf": ("course.pdf", b"%PDF-1.7\ncontent", "application/pdf")},
        headers={"origin": "http://testserver"},
    )

    assert response.status_code == 200
    assert "Upload stored" in response.text
    assert "Python Course" in response.text
    assert "course.pdf" in response.text
    assert "7" in response.text
    assert "Pending" in response.text
    assert "Model processing has not started yet" in response.text
    assert libraries.uploaded_content == b"%PDF-1.7\ncontent"

    package_response = client.get("/library")

    assert "Python Course" in package_response.text
    assert "PDF stored; preparation is pending" in package_response.text
    assert '<span class="package-count">1</span>' in package_response.text


def test_package_upload_rejects_non_pdf_content() -> None:
    response = build_client().post(
        "/package/new",
        data={"name": "Not PDF", "question_count": "5"},
        files={"pdf": ("course.pdf", b"plain text", "application/pdf")},
        headers={"origin": "http://testserver"},
    )

    assert response.status_code == 422
    assert "does not contain a PDF signature" in response.text


def test_package_upload_rejects_duplicate_package_name() -> None:
    response = build_client([ready_package()]).post(
        "/package/new",
        data={"name": "python basics", "question_count": "5"},
        files={"pdf": ("course.pdf", b"%PDF-1.7", "application/pdf")},
        headers={"origin": "http://testserver"},
    )

    assert response.status_code == 422
    assert "Learning package already exists" in response.text


def test_package_upload_rejects_duplicate_pending_name() -> None:
    libraries = StubLibraryManagement()
    preparation_id = UUID("cccccccc-cccc-cccc-cccc-cccccccccccc")
    libraries.preparations.append(
        PackagePreparation(
            id=preparation_id,
            name="Python Course",
            source_filename="first.pdf",
            stored_filename=f"{preparation_id}.pdf",
            question_count=5,
            size_bytes=8,
        )
    )

    response = build_client(libraries=libraries).post(
        "/package/new",
        data={"name": "python course", "question_count": "5"},
        files={"pdf": ("course.pdf", b"%PDF-1.7", "application/pdf")},
        headers={"origin": "http://testserver"},
    )

    assert response.status_code == 422
    assert "Learning package already exists" in response.text


def test_package_upload_rejects_question_count_outside_supported_range() -> None:
    response = build_client().post(
        "/package/new",
        data={"name": "Too many", "question_count": "51"},
        files={"pdf": ("course.pdf", b"%PDF-1.7", "application/pdf")},
        headers={"origin": "http://testserver"},
    )

    assert response.status_code == 422
    assert "Question count must be between 1 and 50" in response.text


def test_package_upload_rejects_files_above_size_limit(
    monkeypatch,
) -> None:
    monkeypatch.setattr(web_application, "MAX_PDF_UPLOAD_BYTES", 8)

    response = build_client().post(
        "/package/new",
        data={"name": "Too large", "question_count": "5"},
        files={"pdf": ("course.pdf", b"%PDF-1.7 extra", "application/pdf")},
        headers={"origin": "http://testserver"},
    )

    assert response.status_code == 422
    assert "PDF file must not exceed 25 MiB" in response.text


def test_package_upload_rejects_cross_origin_submission() -> None:
    response = build_client().post(
        "/package/new",
        data={"name": "Untrusted", "question_count": "5"},
        files={"pdf": ("course.pdf", b"%PDF-1.7", "application/pdf")},
        headers={"origin": "https://attacker.example"},
    )

    assert response.status_code == 403


def test_home_page_lists_the_selected_library() -> None:
    response = build_client().get("/")

    assert response.status_code == 200
    assert "Personal Library" in response.text
    assert "Currently selected" not in response.text
    assert "Open library" not in response.text
    assert 'class="library-button"' in response.text
    assert 'aria-label="Personal Library"' in response.text
    assert "Create and select" not in response.text


def test_library_management_is_on_a_separate_page() -> None:
    response = build_client().get("/libraries/manage")

    assert response.status_code == 200
    assert "Manage libraries" in response.text
    assert ">Create</button>" in response.text
    assert "Create and select" not in response.text


def test_library_management_shows_editor_for_chosen_row() -> None:
    response = build_client().get(
        "/libraries/manage?library_id=11111111-1111-1111-1111-111111111111"
    )

    assert response.status_code == 200
    assert "Edit library" in response.text
    assert "Rename" in response.text
    assert "Delete library" in response.text


def test_library_can_be_renamed_without_changing_its_directory() -> None:
    libraries = StubLibraryManagement()
    response = build_client(libraries=libraries).post(
        "/libraries/rename",
        data={
            "library_id": "11111111-1111-1111-1111-111111111111",
            "name": "Renamed Library",
        },
        headers={"origin": "http://testserver"},
        follow_redirects=False,
    )

    assert response.status_code == 303
    assert libraries.current_library.name == "Renamed Library"
    assert libraries.current_directory == Path("personal-library")


def test_library_deletion_requires_the_exact_display_name() -> None:
    libraries = StubLibraryManagement()
    response = build_client(libraries=libraries).post(
        "/libraries/delete",
        data={
            "library_id": "11111111-1111-1111-1111-111111111111",
            "confirmation": "wrong name",
        },
        headers={"origin": "http://testserver"},
    )

    assert response.status_code == 422
    assert "Library name confirmation does not match" in response.text


def test_last_library_can_be_deleted_and_home_shows_empty_state() -> None:
    libraries = StubLibraryManagement()
    client = build_client(libraries=libraries)

    response = client.post(
        "/libraries/delete",
        data={
            "library_id": "11111111-1111-1111-1111-111111111111",
            "confirmation": "Personal Library",
        },
        headers={"origin": "http://testserver"},
        follow_redirects=False,
    )

    assert response.status_code == 303
    home_response = client.get("/")
    assert "No libraries yet" in home_response.text
    assert "· Packages" not in home_response.text


def test_package_page_redirects_when_no_library_is_open() -> None:
    libraries = StubLibraryManagement()
    libraries.delete_library(
        UUID("11111111-1111-1111-1111-111111111111"),
        confirmation="Personal Library",
        delete_contents=False,
    )

    response = build_client(libraries=libraries).get("/library", follow_redirects=False)

    assert response.status_code == 303
    assert response.headers["location"] == "http://testserver/"


def test_library_can_be_created_and_selected_from_same_origin() -> None:
    libraries = StubLibraryManagement()
    client = build_client(libraries=libraries)

    response = client.post(
        "/libraries",
        data={"name": "German History"},
        headers={"origin": "http://testserver"},
        follow_redirects=False,
    )

    assert response.status_code == 303
    assert response.headers["location"].startswith("http://testserver/libraries/manage?library_id=")
    home_response = client.get("/")
    assert "German History" in home_response.text
    assert libraries.current_directory == Path("personal-library")


def test_existing_library_can_be_selected() -> None:
    libraries = StubLibraryManagement()
    libraries.create_library("German History")
    client = build_client(libraries=libraries)

    response = client.post(
        "/libraries/select",
        data={"library_id": "11111111-1111-1111-1111-111111111111"},
        headers={"origin": "http://testserver"},
        follow_redirects=False,
    )

    assert response.status_code == 303
    assert response.headers["location"] == "http://testserver/library"
    assert libraries.current_directory == Path("personal-library")


def test_library_creation_rejects_a_cross_origin_request() -> None:
    response = build_client().post(
        "/libraries",
        data={"name": "Untrusted"},
        headers={"origin": "https://attacker.example"},
    )

    assert response.status_code == 403


def test_library_creation_shows_validation_errors() -> None:
    response = build_client().post(
        "/libraries",
        data={"name": "  "},
        headers={"origin": "http://testserver"},
    )

    assert response.status_code == 422
    assert "Library name must not be blank" in response.text


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
            StubProgressReporting(),
            libraries=StubLibraryManagement(),
        )
    )

    response = client.get("/package", params={"name": "Python Basics"})

    assert response.status_code == 200
    assert 'aria-label="Main navigation"' in response.text
    assert "Libraries" in response.text
    assert "Manage libraries" in response.text
    assert "Personal Library · Packages" in response.text
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


def test_progress_page_summarizes_attempts_and_learning_focus() -> None:
    last_studied = datetime(2026, 8, 27, 12, 0, tzinfo=UTC)
    next_due = datetime(2026, 8, 29, 12, 0, tzinfo=UTC)
    report = LearningProgressReport(
        package_name="Python Basics",
        total_question_count=5,
        answered_question_count=2,
        due_question_count=1,
        attempt_count=4,
        incorrect_attempt_count=1,
        partially_correct_attempt_count=1,
        correct_attempt_count=2,
        difficult_concepts=(("Closures", 2), ("Decorators", 1)),
        last_studied_at=last_studied,
        next_due_at=next_due,
        unclassified_attempt_count=0,
    )

    response = build_client(
        [ready_package()],
        progress=StubProgressReporting(report),
    ).get("/progress?package=Python%20Basics")

    assert response.status_code == 200
    assert "Learning progress" in response.text
    assert "2/5" in response.text
    assert "40% of the active question bank" in response.text
    assert "50% evaluated as correct" in response.text
    assert "Closures" in response.text
    assert format_local_datetime(last_studied) in response.text
    assert format_local_datetime(next_due) in response.text
    assert last_studied.isoformat() not in response.text


def test_progress_page_explains_an_empty_history() -> None:
    response = build_client(
        [ready_package()],
        progress=StubProgressReporting(),
    ).get("/progress?package=Python%20Basics")

    assert response.status_code == 200
    assert "No difficult concepts recorded." in response.text
    assert "Never" in response.text
    assert "No review scheduled" in response.text


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
