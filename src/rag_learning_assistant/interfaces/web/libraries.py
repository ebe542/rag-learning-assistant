"""Manage selectable local libraries for the loopback web interface."""

import json
import shutil
import sqlite3
from collections.abc import Callable
from contextlib import closing
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import BinaryIO
from uuid import UUID, uuid4

from rag_learning_assistant.application import (
    AnswerEvaluationService,
    DocumentSummaryCatalog,
    DueQuestion,
    LearningPackageCatalog,
    LearningPackageStudyService,
    LearningProgressReport,
    LearningProgressService,
    PackagePreparationService,
    QuestionBankCatalog,
    ReviewScheduler,
    ReviewService,
    StudySessionService,
)
from rag_learning_assistant.generation import (
    HuggingFaceTextGenerator,
    PersistedDocumentSummary,
    SqliteDocumentSummaryRepository,
)
from rag_learning_assistant.learning import (
    LearningPackage,
    PackagePreparation,
    QuestionBank,
    SqliteLearningPackageRepository,
    SqlitePackagePreparationRepository,
    SqliteQuestionBankRepository,
    SqliteQuestionProgressRepository,
    SqliteStudyAttemptRepository,
    StudyAttempt,
)
from rag_learning_assistant.library import SqliteDocumentRepository


@dataclass(frozen=True, slots=True)
class LibraryListItem:
    """Describe one selectable library without exposing mutable state."""

    id: UUID
    name: str
    directory: Path
    has_content: bool


@dataclass(frozen=True, slots=True)
class _LibraryMetadata:
    id: UUID
    name: str


@dataclass(frozen=True, slots=True)
class _LibraryServices:
    packages: LearningPackageCatalog
    preparations: PackagePreparationService
    documents: SqliteDocumentRepository
    summaries: DocumentSummaryCatalog
    questions: QuestionBankCatalog
    study: LearningPackageStudyService
    progress: LearningProgressService


class LocalLibraryManager:
    """Discover sibling libraries and delegate requests to the selected one."""

    def __init__(
        self,
        initial_directory: Path,
        *,
        id_factory: Callable[[], UUID] = uuid4,
        package_remover: Callable[[Path, str], None] | None = None,
    ) -> None:
        self.id_factory = id_factory
        self.package_remover = package_remover
        self.root_directory = initial_directory.resolve().parent
        self.root_directory.mkdir(parents=True, exist_ok=True)
        self._current_directory: Path | None = None
        self._services: _LibraryServices | None = None
        initial_directory = initial_directory.resolve()

        if (initial_directory / "metadata.sqlite3").is_file():
            self._open_directory(initial_directory)

    @property
    def current_directory(self) -> Path | None:
        return self._current_directory

    @property
    def current_library(self) -> LibraryListItem | None:
        if self._current_directory is None:
            return None
        metadata = self._read_or_create_metadata(self._current_directory)
        return LibraryListItem(
            metadata.id,
            metadata.name,
            self._current_directory,
            _library_has_content(self._current_directory),
        )

    def list_libraries(self) -> tuple[LibraryListItem, ...]:
        items = []
        known_ids: set[UUID] = set()
        for directory in self._library_directories():
            metadata = self._read_or_create_metadata(directory)
            if metadata.id in known_ids:
                raise ValueError(f"Duplicate library ID: {metadata.id}")
            known_ids.add(metadata.id)
            items.append(
                LibraryListItem(
                    id=metadata.id,
                    name=metadata.name,
                    directory=directory,
                    has_content=_library_has_content(directory),
                )
            )
        return tuple(sorted(items, key=lambda item: (item.name.casefold(), str(item.id))))

    def create_library(self, name: str) -> LibraryListItem:
        normalized_name = _validate_library_name(name)
        if any(
            item.name.casefold() == normalized_name.casefold() for item in self.list_libraries()
        ):
            raise ValueError(f"Library already exists: {normalized_name}")

        library_id = self._next_library_id()
        directory = self.root_directory / str(library_id)
        directory.mkdir()
        self._write_metadata(directory, _LibraryMetadata(library_id, normalized_name))
        SqliteLearningPackageRepository(directory / "metadata.sqlite3")
        return LibraryListItem(library_id, normalized_name, directory.resolve(), False)

    def select_library(self, library_id: UUID) -> LibraryListItem:
        matching_library = next(
            (item for item in self.list_libraries() if item.id == library_id),
            None,
        )
        if matching_library is None:
            raise LookupError(f"Library not found: {library_id}")

        self._current_directory = matching_library.directory.resolve()
        self._services = self._build_services(self._current_directory)
        return LibraryListItem(
            matching_library.id,
            matching_library.name,
            self._current_directory,
            matching_library.has_content,
        )

    def rename_library(self, library_id: UUID, name: str) -> LibraryListItem:
        normalized_name = _validate_library_name(name)
        libraries = self.list_libraries()
        matching_library = next((item for item in libraries if item.id == library_id), None)
        if matching_library is None:
            raise LookupError(f"Library not found: {library_id}")
        if any(
            item.id != library_id and item.name.casefold() == normalized_name.casefold()
            for item in libraries
        ):
            raise ValueError(f"Library already exists: {normalized_name}")

        self._write_metadata(
            matching_library.directory,
            _LibraryMetadata(library_id, normalized_name),
        )
        return LibraryListItem(
            library_id,
            normalized_name,
            matching_library.directory,
            matching_library.has_content,
        )

    def delete_library(
        self,
        library_id: UUID,
        *,
        confirmation: str,
        delete_contents: bool,
    ) -> None:
        libraries = self.list_libraries()
        matching_library = next((item for item in libraries if item.id == library_id), None)
        if matching_library is None:
            raise LookupError(f"Library not found: {library_id}")
        if confirmation != matching_library.name:
            raise ValueError("Library name confirmation does not match")
        if matching_library.has_content and not delete_contents:
            raise ValueError("Confirm deletion of all library contents")
        if _library_has_active_preparation(matching_library.directory):
            raise ValueError("Library is currently preparing a package and cannot be deleted")

        target = matching_library.directory.resolve()
        if target.parent != self.root_directory or target == self.root_directory:
            raise ValueError("Library directory is outside the configured workspace")
        metadata = self._read_or_create_metadata(target)
        if metadata.id != library_id:
            raise ValueError("Library identity changed before deletion")

        if target == self._current_directory:
            replacement = next((item for item in libraries if item.id != library_id), None)
            if replacement is None:
                self._current_directory = None
                self._services = None
            else:
                self._open_directory(replacement.directory)
        shutil.rmtree(target)

    def _library_directories(self) -> list[Path]:
        return sorted(
            (
                path
                for path in self.root_directory.iterdir()
                if path.is_dir()
                and path.resolve().parent == self.root_directory
                and (path / "metadata.sqlite3").is_file()
            ),
            key=lambda path: path.name.casefold(),
        )

    def _open_directory(self, directory: Path) -> None:
        resolved_directory = directory.resolve()
        self._services = self._build_services(resolved_directory)
        self._current_directory = resolved_directory
        self._read_or_create_metadata(resolved_directory)

    def _require_services(self) -> _LibraryServices:
        if self._services is None:
            raise RuntimeError("No library is open")
        return self._services

    def _next_library_id(self) -> UUID:
        known_ids = {item.id for item in self.list_libraries()}
        while (library_id := self.id_factory()) in known_ids or (
            self.root_directory / str(library_id)
        ).exists():
            continue
        return library_id

    def _read_or_create_metadata(self, directory: Path) -> _LibraryMetadata:
        metadata_path = directory / "library.json"
        if not metadata_path.is_file():
            metadata = _LibraryMetadata(self._next_legacy_id(directory), directory.name)
            self._write_metadata(directory, metadata)
            return metadata

        try:
            payload = json.loads(metadata_path.read_text(encoding="utf-8"))
            return _LibraryMetadata(
                id=UUID(payload["id"]),
                name=_validate_library_name(payload["name"]),
            )
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
            raise ValueError(f"Invalid library metadata: {metadata_path}") from error

    def _next_legacy_id(self, directory: Path) -> UUID:
        library_id = self.id_factory()
        metadata_paths = self.root_directory.glob("*/library.json")
        known_ids = set()
        for metadata_path in metadata_paths:
            if metadata_path.parent == directory:
                continue
            try:
                known_ids.add(UUID(json.loads(metadata_path.read_text(encoding="utf-8"))["id"]))
            except (KeyError, TypeError, ValueError, json.JSONDecodeError):
                continue
        while library_id in known_ids:
            library_id = self.id_factory()
        return library_id

    @staticmethod
    def _write_metadata(directory: Path, metadata: _LibraryMetadata) -> None:
        payload = json.dumps(
            {"id": str(metadata.id), "name": metadata.name},
            ensure_ascii=False,
            indent=2,
        )
        temporary_path = directory / "library.json.tmp"
        temporary_path.write_text(f"{payload}\n", encoding="utf-8")
        temporary_path.replace(directory / "library.json")

    def list_packages(self) -> list[LearningPackage]:
        return self._require_services().packages.list_packages()

    def list_package_preparations(self) -> list[PackagePreparation]:
        return self._require_services().preparations.list_all()

    def store_package_upload(
        self,
        *,
        name: str,
        source_filename: str,
        question_count: int,
        size_bytes: int,
        content_sha256: str,
        source: BinaryIO,
    ) -> PackagePreparation:
        duplicate_document = self._require_services().documents.find_by_content_hash(content_sha256)
        if duplicate_document is not None:
            raise ValueError(
                f"This PDF already exists in the library as {duplicate_document.source}"
            )
        return self._require_services().preparations.store(
            name=name,
            source_filename=source_filename,
            question_count=question_count,
            size_bytes=size_bytes,
            content_sha256=content_sha256,
            source=source,
        )

    def retry_package_preparation(self, preparation_id: UUID) -> PackagePreparation:
        return self._require_services().preparations.retry_failed(
            preparation_id,
            now=datetime.now(UTC),
        )

    def delete_failed_package_preparation(
        self,
        preparation_id: UUID,
    ) -> PackagePreparation:
        services = self._require_services()
        preparation = next(
            (item for item in services.preparations.list_all() if item.id == preparation_id),
            None,
        )
        if preparation is None:
            raise ValueError(f"Package preparation does not exist: {preparation_id}")
        materialized = next(
            (
                package
                for package in services.packages.list_packages()
                if package.name.casefold() == preparation.name.casefold()
            ),
            None,
        )
        if materialized is not None:
            if self.package_remover is None:
                raise RuntimeError("Package removal service is not configured")
            assert self._current_directory is not None
            self.package_remover(self._current_directory, materialized.name)
        return services.preparations.delete_failed(preparation_id)

    def rename_package(self, name: str, new_name: str) -> LearningPackage:
        """Rename a materialized package without changing its derived content."""

        normalized_name = new_name.strip()
        if not normalized_name:
            raise ValueError("Learning package name must not be blank")
        if len(normalized_name) > 100:
            raise ValueError("Learning package name must not exceed 100 characters")
        services = self._require_services()
        package = next(
            (
                item
                for item in services.packages.list_packages()
                if item.name.casefold() == name.casefold()
            ),
            None,
        )
        if package is None:
            raise ValueError(f"Learning package does not exist: {name}")
        renamed = replace(package, name=normalized_name)
        assert self._current_directory is not None
        SqliteLearningPackageRepository(self._current_directory / "metadata.sqlite3").replace(
            renamed
        )
        return renamed

    def delete_package(self, name: str, *, confirmation: str) -> None:
        """Delete one package through the shared full removal lifecycle."""

        package = self.get_package(name)
        if confirmation != package.name:
            raise ValueError("Package name confirmation does not match")
        if self.package_remover is None:
            raise RuntimeError("Package removal service is not configured")
        assert self._current_directory is not None
        self.package_remover(self._current_directory, package.name)

    def get_package(self, name: str) -> LearningPackage:
        return self._require_services().packages.get_package(name)

    def get_document_summary(
        self,
        document_id: UUID,
        identity_fingerprint: str,
    ) -> PersistedDocumentSummary:
        return self._require_services().summaries.get_document_summary(
            document_id,
            identity_fingerprint,
        )

    def get_document_bank(
        self,
        document_id: UUID,
        identity_fingerprint: str,
    ) -> QuestionBank:
        return self._require_services().questions.get_document_bank(
            document_id,
            identity_fingerprint,
        )

    def next_due(self, package_name: str, *, as_of: datetime) -> DueQuestion | None:
        return self._require_services().study.next_due(package_name, as_of=as_of)

    def record_answer(
        self,
        package_name: str,
        question_number: int,
        *,
        answer_text: str,
        answered_at: datetime,
    ) -> StudyAttempt:
        return self._require_services().study.record_answer(
            package_name,
            question_number,
            answer_text=answer_text,
            answered_at=answered_at,
        )

    def report(self, package_name: str, *, as_of: datetime) -> LearningProgressReport:
        return self._require_services().progress.report(package_name, as_of=as_of)

    @staticmethod
    def _build_services(library_directory: Path) -> _LibraryServices:
        database_path = library_directory / "metadata.sqlite3"
        documents = SqliteDocumentRepository(database_path)
        package_repository = SqliteLearningPackageRepository(database_path)
        preparations = PackagePreparationService(
            SqlitePackagePreparationRepository(database_path),
            library_directory / "uploads",
        )
        preparations.backfill_missing_hashes()
        progress_repository = SqliteQuestionProgressRepository(database_path)
        attempt_repository = SqliteStudyAttemptRepository(database_path)
        packages = LearningPackageCatalog(package_repository)
        summaries = DocumentSummaryCatalog(
            documents,
            SqliteDocumentSummaryRepository(database_path),
        )
        questions = QuestionBankCatalog(
            documents,
            SqliteQuestionBankRepository(database_path),
        )
        reviewer = ReviewService(
            banks=questions,
            progress=progress_repository,
            scheduler=ReviewScheduler(),
        )
        study = LearningPackageStudyService(
            packages=package_repository,
            sessions=StudySessionService(
                banks=questions,
                reviewer=reviewer,
                attempts=attempt_repository,
                attempt_id_factory=uuid4,
                evaluator=AnswerEvaluationService(HuggingFaceTextGenerator()),
            ),
        )
        progress = LearningProgressService(
            packages=package_repository,
            banks=questions,
            progress=progress_repository,
            attempts=attempt_repository,
        )
        return _LibraryServices(
            packages,
            preparations,
            documents,
            summaries,
            questions,
            study,
            progress,
        )


def _validate_library_name(name: str) -> str:
    normalized_name = name.strip()
    if not 1 <= len(normalized_name) <= 100:
        raise ValueError("Library name must contain between 1 and 100 characters")
    if any(ord(character) < 32 for character in normalized_name):
        raise ValueError("Library name must not contain control characters")
    return normalized_name


def _library_has_content(directory: Path) -> bool:
    """Detect persisted user data without treating schemas as content."""

    vectors_path = directory / "vectors.faiss"
    if vectors_path.is_file() and vectors_path.stat().st_size > 0:
        return True

    database_path = directory / "metadata.sqlite3"
    if not database_path.is_file():
        return False
    with closing(sqlite3.connect(database_path)) as connection:
        table_names = (
            row[0]
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table' AND name NOT LIKE 'sqlite_%'"
            )
        )
        for table_name in table_names:
            escaped_name = table_name.replace('"', '""')
            if connection.execute(
                f'SELECT EXISTS(SELECT 1 FROM "{escaped_name}" LIMIT 1)'
            ).fetchone()[0]:
                return True
    return False


def _library_has_active_preparation(directory: Path) -> bool:
    database_path = directory / "metadata.sqlite3"
    if not database_path.is_file():
        return False
    with closing(sqlite3.connect(database_path)) as connection:
        table_exists = connection.execute(
            "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'package_preparations'"
        ).fetchone()
        if table_exists is None:
            return False
        return (
            connection.execute(
                """
                SELECT 1 FROM package_preparations
                WHERE status IN ('indexing', 'summarizing', 'generating_questions')
                LIMIT 1
                """
            ).fetchone()
            is not None
        )
