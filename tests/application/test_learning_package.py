from pathlib import Path
from uuid import UUID

import pytest

from rag_learning_assistant.application import (
    LearningPackageCatalog,
    LearningPackageService,
)
from rag_learning_assistant.learning import (
    LearningPackage,
    LearningPackageStatus,
)
from rag_learning_assistant.library import IndexedDocument


class RecordingPackageRepository:
    def __init__(
        self,
        existing: LearningPackage | None = None,
    ) -> None:
        self.existing = existing
        self.saved: list[LearningPackage] = []
        self.replaced: list[LearningPackage] = []

    def find_by_name(self, name: str) -> LearningPackage | None:
        if self.existing is not None and self.existing.name.casefold() == name.casefold():
            return self.existing

        return None

    def list_all(self) -> list[LearningPackage]:
        if self.existing is None:
            return []

        return [self.existing]

    def save(self, package: LearningPackage) -> None:
        self.saved.append(package)

    def replace(self, package: LearningPackage) -> None:
        self.replaced.append(package)


class RecordingDocumentImporter:
    def __init__(self, document: IndexedDocument) -> None:
        self.document = document
        self.paths: list[Path] = []
        self.removed_document_ids: list[UUID] = []

    def add_document(self, path: Path) -> IndexedDocument:
        self.paths.append(path)
        return self.document

    def remove_document(self, document_id: UUID) -> IndexedDocument:
        self.removed_document_ids.append(document_id)
        return self.document


class FailingSummaryPreparer:
    def prepare_summary(self, document_id: UUID) -> str:
        raise RuntimeError("Summary generation failed")


class UnusedQuestionPreparer:
    def prepare_questions(
        self,
        document_id: UUID,
        summary_identity_fingerprint: str,
        *,
        question_count: int,
    ) -> str:
        raise AssertionError("Questions must not be generated")


def test_prepare_preserves_indexed_checkpoint_when_summary_fails() -> None:
    document = IndexedDocument(
        id=UUID("bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb"),
        source="python-book.pdf",
        content_sha256="c" * 64,
        page_count=100,
        chunk_count=200,
    )
    packages = RecordingPackageRepository()
    documents = RecordingDocumentImporter(document)
    service = LearningPackageService(
        packages=packages,
        documents=documents,
        summaries=FailingSummaryPreparer(),
        questions=UnusedQuestionPreparer(),
        id_factory=lambda: UUID("aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa"),
    )

    with pytest.raises(RuntimeError, match="Summary generation failed"):
        service.prepare(
            name="Python Basics",
            pdf_path=Path("python-book.pdf"),
            question_count=20,
        )

    assert documents.paths == [Path("python-book.pdf")]
    assert packages.saved == [
        LearningPackage(
            id=UUID("aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa"),
            name="Python Basics",
            document_id=document.id,
            status=LearningPackageStatus.INDEXED,
        )
    ]


class RecordingSummaryPreparer:
    def __init__(self) -> None:
        self.document_ids: list[UUID] = []

    def prepare_summary(self, document_id: UUID) -> str:
        self.document_ids.append(document_id)
        return "d" * 64


class RecordingQuestionPreparer:
    def __init__(self) -> None:
        self.calls: list[tuple[UUID, str, int]] = []

    def prepare_questions(
        self,
        document_id: UUID,
        summary_identity_fingerprint: str,
        *,
        question_count: int,
    ) -> str:
        self.calls.append(
            (
                document_id,
                summary_identity_fingerprint,
                question_count,
            )
        )
        return "e" * 64


def test_prepare_creates_ready_learning_package() -> None:
    document = IndexedDocument(
        id=UUID("bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb"),
        source="python-book.pdf",
        content_sha256="c" * 64,
        page_count=100,
        chunk_count=200,
    )
    packages = RecordingPackageRepository()
    documents = RecordingDocumentImporter(document)
    summaries = RecordingSummaryPreparer()
    questions = RecordingQuestionPreparer()

    phases: list[str] = []

    service = LearningPackageService(
        packages=packages,
        documents=documents,
        summaries=summaries,
        questions=questions,
        id_factory=lambda: UUID("aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa"),
        progress=phases.append,
    )

    result = service.prepare(
        name="Python Basics",
        pdf_path=Path("python-book.pdf"),
        question_count=20,
    )

    assert phases == [
        "index",
        "summarize",
        "questions",
        "ready",
    ]
    assert result == LearningPackage(
        id=UUID("aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa"),
        name="Python Basics",
        document_id=document.id,
        status=LearningPackageStatus.READY,
        summary_identity_fingerprint="d" * 64,
        question_bank_identity_fingerprint="e" * 64,
    )
    assert documents.paths == [Path("python-book.pdf")]
    assert summaries.document_ids == [document.id]
    assert questions.calls == [
        (document.id, "d" * 64, 20),
    ]
    assert [package.status for package in packages.replaced] == [
        LearningPackageStatus.SUMMARIZED,
        LearningPackageStatus.READY,
    ]


def test_prepare_resumes_summarized_learning_package() -> None:
    document_id = UUID("bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb")
    existing = LearningPackage(
        id=UUID("aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa"),
        name="Python Basics",
        document_id=document_id,
        status=LearningPackageStatus.SUMMARIZED,
        summary_identity_fingerprint="d" * 64,
    )
    packages = RecordingPackageRepository(existing)
    documents = RecordingDocumentImporter(
        IndexedDocument(
            id=document_id,
            source="python-book.pdf",
            content_sha256="c" * 64,
            page_count=100,
            chunk_count=200,
        )
    )
    summaries = RecordingSummaryPreparer()
    questions = RecordingQuestionPreparer()

    phases: list[str] = []

    service = LearningPackageService(
        packages=packages,
        documents=documents,
        summaries=summaries,
        questions=questions,
        progress=phases.append,
    )

    result = service.prepare(
        name="python basics",
        pdf_path=Path("ignored.pdf"),
        question_count=20,
    )

    assert phases == [
        "questions",
        "ready",
    ]
    assert result.status is LearningPackageStatus.READY
    assert result.question_bank_identity_fingerprint == "e" * 64
    assert documents.paths == []
    assert summaries.document_ids == []
    assert questions.calls == [
        (document_id, "d" * 64, 20),
    ]
    assert packages.saved == []
    assert packages.replaced == [result]


def test_prepare_reuses_ready_learning_package() -> None:
    document_id = UUID("bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb")
    existing = LearningPackage(
        id=UUID("aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa"),
        name="Python Basics",
        document_id=document_id,
        status=LearningPackageStatus.READY,
        summary_identity_fingerprint="d" * 64,
        question_bank_identity_fingerprint="e" * 64,
    )
    packages = RecordingPackageRepository(existing)
    documents = RecordingDocumentImporter(
        IndexedDocument(
            id=document_id,
            source="python-book.pdf",
            content_sha256="c" * 64,
            page_count=100,
            chunk_count=200,
        )
    )
    summaries = RecordingSummaryPreparer()
    questions = RecordingQuestionPreparer()
    service = LearningPackageService(
        packages=packages,
        documents=documents,
        summaries=summaries,
        questions=questions,
    )

    result = service.prepare(
        name="Python Basics",
        pdf_path=Path("ignored.pdf"),
        question_count=20,
    )

    assert result == existing
    assert documents.paths == []
    assert summaries.document_ids == []
    assert questions.calls == []
    assert packages.saved == []
    assert packages.replaced == []


def test_remove_learning_package_removes_its_source_document() -> None:
    document_id = UUID("bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb")
    package = LearningPackage(
        id=UUID("aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa"),
        name="Python Basics",
        document_id=document_id,
        status=LearningPackageStatus.READY,
        summary_identity_fingerprint="d" * 64,
        question_bank_identity_fingerprint="e" * 64,
    )
    documents = RecordingDocumentImporter(
        IndexedDocument(
            id=document_id,
            source="python-book.pdf",
            content_sha256="c" * 64,
            page_count=100,
            chunk_count=200,
        )
    )
    service = LearningPackageService(
        packages=RecordingPackageRepository(package),
        documents=documents,
        summaries=RecordingSummaryPreparer(),
        questions=RecordingQuestionPreparer(),
    )

    assert service.remove("python basics") == package
    assert documents.removed_document_ids == [document_id]


def test_prepare_resumes_indexed_learning_package() -> None:
    document_id = UUID("bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb")
    existing = LearningPackage(
        id=UUID("aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa"),
        name="Python Basics",
        document_id=document_id,
        status=LearningPackageStatus.INDEXED,
    )
    packages = RecordingPackageRepository(existing)
    documents = RecordingDocumentImporter(
        IndexedDocument(
            id=document_id,
            source="python-book.pdf",
            content_sha256="c" * 64,
            page_count=100,
            chunk_count=200,
        )
    )
    summaries = RecordingSummaryPreparer()
    questions = RecordingQuestionPreparer()
    service = LearningPackageService(
        packages=packages,
        documents=documents,
        summaries=summaries,
        questions=questions,
    )

    result = service.prepare(
        name="Python Basics",
        pdf_path=Path("ignored.pdf"),
        question_count=20,
    )

    assert result.status is LearningPackageStatus.READY
    assert documents.paths == []
    assert summaries.document_ids == [document_id]
    assert questions.calls == [
        (document_id, "d" * 64, 20),
    ]
    assert [package.status for package in packages.replaced] == [
        LearningPackageStatus.SUMMARIZED,
        LearningPackageStatus.READY,
    ]


@pytest.mark.parametrize("name", ["", "   "])
def test_prepare_rejects_blank_name_before_import(name: str) -> None:
    document = IndexedDocument(
        id=UUID("bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb"),
        source="python-book.pdf",
        content_sha256="c" * 64,
        page_count=100,
        chunk_count=200,
    )
    documents = RecordingDocumentImporter(document)
    service = LearningPackageService(
        packages=RecordingPackageRepository(),
        documents=documents,
        summaries=RecordingSummaryPreparer(),
        questions=RecordingQuestionPreparer(),
    )

    with pytest.raises(
        ValueError,
        match="Learning package name must not be blank",
    ):
        service.prepare(
            name=name,
            pdf_path=Path("python-book.pdf"),
            question_count=20,
        )

    assert documents.paths == []


@pytest.mark.parametrize("question_count", [0, -1])
def test_prepare_rejects_invalid_question_count_before_import(
    question_count: int,
) -> None:
    document = IndexedDocument(
        id=UUID("bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb"),
        source="python-book.pdf",
        content_sha256="c" * 64,
        page_count=100,
        chunk_count=200,
    )
    documents = RecordingDocumentImporter(document)
    service = LearningPackageService(
        packages=RecordingPackageRepository(),
        documents=documents,
        summaries=RecordingSummaryPreparer(),
        questions=RecordingQuestionPreparer(),
    )

    with pytest.raises(
        ValueError,
        match="question_count must be positive",
    ):
        service.prepare(
            name="Python Basics",
            pdf_path=Path("python-book.pdf"),
            question_count=question_count,
        )

    assert documents.paths == []


class StaticPackageRepository:
    def __init__(
        self,
        packages: list[LearningPackage],
    ) -> None:
        self.packages = packages

    def list_all(self) -> list[LearningPackage]:
        return list(self.packages)


def test_learning_package_catalog_lists_available_packages() -> None:
    packages = [
        LearningPackage(
            id=UUID("aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa"),
            name="Algorithms",
            document_id=UUID("bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb"),
            status=LearningPackageStatus.INDEXED,
        ),
        LearningPackage(
            id=UUID("cccccccc-cccc-cccc-cccc-cccccccccccc"),
            name="Python Basics",
            document_id=UUID("dddddddd-dddd-dddd-dddd-dddddddddddd"),
            status=LearningPackageStatus.SUMMARIZED,
            summary_identity_fingerprint="e" * 64,
        ),
    ]
    catalog = LearningPackageCatalog(StaticPackageRepository(packages))

    assert catalog.list_packages() == packages
