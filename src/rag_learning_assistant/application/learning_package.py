"""Prepare user-facing learning packages from source documents."""

from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from typing import Protocol
from uuid import UUID, uuid4

from rag_learning_assistant.learning import (
    LearningPackage,
    LearningPackageStatus,
)
from rag_learning_assistant.library import IndexedDocument

from .package_study import LearningPackageNotFoundError


class LearningPackageReader(Protocol):
    """Read the user-facing learning packages of one library."""

    def find_by_name(self, name: str) -> LearningPackage | None: ...

    def list_all(self) -> list[LearningPackage]: ...


class LearningPackageRepository(
    LearningPackageReader,
    Protocol,
):
    """Persist the active preparation state of learning packages."""

    def find_by_name(
        self,
        name: str,
    ) -> LearningPackage | None: ...

    def save(self, package: LearningPackage) -> None: ...

    def save_from_preparation(
        self,
        package: LearningPackage,
        preparation_id: UUID,
    ) -> None: ...

    def replace(self, package: LearningPackage) -> None: ...

    def is_name_reserved(self, name: str) -> bool: ...


class PackageDocumentImporter(Protocol):
    """Import one source document into the persistent library."""

    def add_document(
        self,
        path: Path,
        *,
        source_name: str | None = None,
    ) -> IndexedDocument: ...

    def remove_document(self, document_id: UUID) -> IndexedDocument: ...


class PackageSummaryPreparer(Protocol):
    """Prepare one persisted summary and return its identity."""

    def prepare_summary(self, document_id: UUID) -> str: ...


class PackageQuestionPreparer(Protocol):
    """Prepare one persisted question bank and return its identity."""

    def prepare_questions(
        self,
        document_id: UUID,
        summary_identity_fingerprint: str,
        *,
        question_count: int,
    ) -> str: ...


class LearningPackageCatalog:
    """Provide read-only access to available learning packages."""

    def __init__(
        self,
        packages: LearningPackageReader,
    ) -> None:
        self.packages = packages

    def list_packages(self) -> list[LearningPackage]:
        """Return packages in repository-defined display order."""

        return self.packages.list_all()

    def get_package(self, name: str) -> LearningPackage:
        """Return one package selected by its user-facing name."""

        package = self.packages.find_by_name(name)
        if package is None:
            raise LearningPackageNotFoundError(f"Learning package not found: {name}")
        return package


class LearningPackageService:
    """Coordinate document preparation through resumable checkpoints."""

    def __init__(
        self,
        packages: LearningPackageRepository,
        documents: PackageDocumentImporter,
        summaries: PackageSummaryPreparer,
        questions: PackageQuestionPreparer,
        id_factory: Callable[[], UUID] = uuid4,
        progress: Callable[[str], None] | None = None,
    ) -> None:
        self.packages = packages
        self.documents = documents
        self.summaries = summaries
        self.questions = questions
        self.id_factory = id_factory
        self.progress = progress

    def prepare(
        self,
        *,
        name: str,
        pdf_path: Path,
        question_count: int,
        preparation_id: UUID | None = None,
        source_filename: str | None = None,
    ) -> LearningPackage:
        """Prepare or resume one learning package by its user-facing name."""

        if not name.strip():
            raise ValueError("Learning package name must not be blank")

        if question_count < 1:
            raise ValueError("question_count must be positive")

        package = self.packages.find_by_name(name)

        if package is None:
            if preparation_id is None and self.packages.is_name_reserved(name):
                raise ValueError(f"Learning package already exists: {name}")
            self._report_progress("index")
            document = (
                self.documents.add_document(pdf_path)
                if source_filename is None
                else self.documents.add_document(pdf_path, source_name=source_filename)
            )
            package = LearningPackage(
                id=self.id_factory(),
                name=name,
                document_id=document.id,
                status=LearningPackageStatus.INDEXED,
            )
            # Persist each expensive completed phase before starting the next
            # one, so a later retry can resume without repeating earlier work.
            if preparation_id is None:
                self.packages.save(package)
            else:
                self.packages.save_from_preparation(package, preparation_id)

        if package.status is LearningPackageStatus.INDEXED:
            self._report_progress("summarize")
            summary_identity = self.summaries.prepare_summary(package.document_id)
            package = replace(
                package,
                status=LearningPackageStatus.SUMMARIZED,
                summary_identity_fingerprint=summary_identity,
            )
            self.packages.replace(package)

        if package.status is LearningPackageStatus.SUMMARIZED:
            if package.summary_identity_fingerprint is None:
                raise RuntimeError("Summarized learning package has no summary identity")
            self._report_progress("questions")
            question_bank_identity = self.questions.prepare_questions(
                package.document_id,
                package.summary_identity_fingerprint,
                question_count=question_count,
            )
            package = replace(
                package,
                status=LearningPackageStatus.READY,
                question_bank_identity_fingerprint=(question_bank_identity),
            )
            self.packages.replace(package)

        self._report_progress("ready")
        return package

    def remove(self, name: str) -> LearningPackage:
        """Remove a package and all data derived from its source document."""

        package = self.packages.find_by_name(name)
        if package is None:
            raise LearningPackageNotFoundError(f"Learning package not found: {name}")

        self.documents.remove_document(package.document_id)
        return package

    def _report_progress(self, phase: str) -> None:
        """Report product progress without coupling it to one interface."""

        if self.progress is not None:
            self.progress(phase)
