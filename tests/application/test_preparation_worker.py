from io import BytesIO
from pathlib import Path
from uuid import UUID

from rag_learning_assistant.application import (
    LearningPackageService,
    PackagePreparationService,
    PackagePreparationWorker,
)
from rag_learning_assistant.learning import (
    LearningLanguage,
    LearningPackageStatus,
    PackagePreparation,
    PackagePreparationStatus,
    SqliteLearningPackageRepository,
    SqlitePackagePreparationRepository,
)
from rag_learning_assistant.library import IndexedDocument


class StubDocumentImporter:
    def add_document(
        self,
        path: Path,
        *,
        source_name: str | None = None,
    ) -> IndexedDocument:
        assert path.read_bytes() == b"%PDF-1.7"
        assert source_name == "course.pdf"
        return IndexedDocument(
            id=UUID("bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb"),
            source=source_name,
            content_sha256="c" * 64,
            page_count=1,
            chunk_count=1,
        )

    def remove_document(self, document_id: UUID) -> IndexedDocument:
        raise AssertionError("Worker must not remove its document")


class StubSummaryPreparer:
    def prepare_summary(
        self,
        document_id: UUID,
        *,
        learning_language: LearningLanguage,
    ) -> str:
        return "d" * 64


class FailingSummaryPreparer:
    def prepare_summary(
        self,
        document_id: UUID,
        *,
        learning_language: LearningLanguage,
    ) -> str:
        raise RuntimeError("Summary failed")


class StubQuestionPreparer:
    def prepare_questions(
        self,
        document_id: UUID,
        summary_identity_fingerprint: str,
        *,
        question_count: int,
    ) -> str:
        assert question_count == 7
        return "e" * 64


def build_preparation(
    tmp_path: Path,
) -> tuple[
    PackagePreparationService,
    SqliteLearningPackageRepository,
    Path,
]:
    database_path = tmp_path / "metadata.sqlite3"
    uploads = tmp_path / "uploads"
    preparations = PackagePreparationService(
        SqlitePackagePreparationRepository(database_path),
        uploads,
        id_factory=lambda: UUID("11111111-1111-1111-1111-111111111111"),
    )
    preparations.store(
        name="Python Course",
        source_filename="course.pdf",
        question_count=7,
        size_bytes=8,
        content_sha256="a" * 64,
        source=BytesIO(b"%PDF-1.7"),
    )
    return preparations, SqliteLearningPackageRepository(database_path), uploads


def test_worker_processes_one_pending_upload_to_ready_package(tmp_path: Path) -> None:
    preparations, packages, uploads = build_preparation(tmp_path)

    worker = PackagePreparationWorker(
        preparations,
        lambda progress: LearningPackageService(
            packages=packages,
            documents=StubDocumentImporter(),
            summaries=StubSummaryPreparer(),
            questions=StubQuestionPreparer(),
            id_factory=lambda: UUID("aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa"),
            progress=progress,
        ),
        uploads,
        token_factory=lambda: UUID("ffffffff-ffff-ffff-ffff-ffffffffffff"),
        heartbeat_interval=3600,
    )

    assert worker.run_once() is True

    package = packages.find_by_name("Python Course")
    assert package is not None
    assert package.status is LearningPackageStatus.READY
    assert preparations.list_all() == []
    assert list(uploads.iterdir()) == []
    assert worker.run_once() is False


def test_failed_worker_attempt_can_resume_from_indexed_checkpoint(tmp_path: Path) -> None:
    preparations, packages, uploads = build_preparation(tmp_path)
    package_id = UUID("aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa")
    lease_token = UUID("ffffffff-ffff-ffff-ffff-ffffffffffff")
    reported_failures: list[tuple[Exception, PackagePreparation]] = []
    failing_worker = PackagePreparationWorker(
        preparations,
        lambda progress: LearningPackageService(
            packages=packages,
            documents=StubDocumentImporter(),
            summaries=FailingSummaryPreparer(),
            questions=StubQuestionPreparer(),
            id_factory=lambda: package_id,
            progress=progress,
        ),
        uploads,
        token_factory=lambda: lease_token,
        heartbeat_interval=3600,
        failure_reporter=lambda error, preparation: reported_failures.append((error, preparation)),
    )

    assert failing_worker.run_once() is True
    failed = preparations.list_all()[0]
    assert failed.status is PackagePreparationStatus.FAILED
    assert failed.failure_message == "RuntimeError: Summary failed"
    assert len(reported_failures) == 1
    assert isinstance(reported_failures[0][0], RuntimeError)
    assert reported_failures[0][1].name == "Python Course"
    indexed_package = packages.find_by_name("Python Course")
    assert indexed_package is not None
    assert indexed_package.status is LearningPackageStatus.INDEXED
    assert failed.updated_at is not None
    preparations.retry_failed(failed.id, now=failed.updated_at)

    resumed_worker = PackagePreparationWorker(
        preparations,
        lambda progress: LearningPackageService(
            packages=packages,
            documents=StubDocumentImporter(),
            summaries=StubSummaryPreparer(),
            questions=StubQuestionPreparer(),
            id_factory=lambda: package_id,
            progress=progress,
        ),
        uploads,
        token_factory=lambda: lease_token,
        heartbeat_interval=3600,
    )
    assert resumed_worker.run_once() is True
    ready_package = packages.find_by_name("Python Course")
    assert ready_package is not None
    assert ready_package.status is LearningPackageStatus.READY
    assert preparations.list_all() == []
