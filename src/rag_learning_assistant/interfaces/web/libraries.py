"""Manage selectable local libraries for the loopback web interface."""

import json
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from uuid import UUID, uuid4

from rag_learning_assistant.application import (
    AnswerEvaluationService,
    DocumentSummaryCatalog,
    DueQuestion,
    LearningPackageCatalog,
    LearningPackageStudyService,
    LearningProgressReport,
    LearningProgressService,
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
    QuestionBank,
    SqliteLearningPackageRepository,
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


@dataclass(frozen=True, slots=True)
class _LibraryMetadata:
    id: UUID
    name: str


@dataclass(frozen=True, slots=True)
class _LibraryServices:
    packages: LearningPackageCatalog
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
    ) -> None:
        self.id_factory = id_factory
        self.root_directory = initial_directory.resolve().parent
        self._current_directory = initial_directory.resolve()
        self._services = self._build_services(self._current_directory)
        self._read_or_create_metadata(self._current_directory)

    @property
    def current_directory(self) -> Path:
        return self._current_directory

    @property
    def current_library(self) -> LibraryListItem:
        metadata = self._read_or_create_metadata(self._current_directory)
        return LibraryListItem(metadata.id, metadata.name, self._current_directory)

    def list_libraries(self) -> tuple[LibraryListItem, ...]:
        library_directories = (
            path
            for path in self.root_directory.iterdir()
            if path.is_dir() and (path / "metadata.sqlite3").is_file()
        )
        items = []
        known_ids: set[UUID] = set()
        for directory in library_directories:
            metadata = self._read_or_create_metadata(directory)
            if metadata.id in known_ids:
                raise ValueError(f"Duplicate library ID: {metadata.id}")
            known_ids.add(metadata.id)
            items.append(
                LibraryListItem(
                    id=metadata.id,
                    name=metadata.name,
                    directory=directory,
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
        return LibraryListItem(library_id, normalized_name, directory.resolve())

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
        )

    def _next_library_id(self) -> UUID:
        known_ids = {item.id for item in self.list_libraries()}
        while (library_id := self.id_factory()) in known_ids:
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
        return self._services.packages.list_packages()

    def get_package(self, name: str) -> LearningPackage:
        return self._services.packages.get_package(name)

    def get_document_summary(
        self,
        document_id: UUID,
        identity_fingerprint: str,
    ) -> PersistedDocumentSummary:
        return self._services.summaries.get_document_summary(
            document_id,
            identity_fingerprint,
        )

    def get_document_bank(
        self,
        document_id: UUID,
        identity_fingerprint: str,
    ) -> QuestionBank:
        return self._services.questions.get_document_bank(
            document_id,
            identity_fingerprint,
        )

    def next_due(self, package_name: str, *, as_of: datetime) -> DueQuestion | None:
        return self._services.study.next_due(package_name, as_of=as_of)

    def record_answer(
        self,
        package_name: str,
        question_number: int,
        *,
        answer_text: str,
        answered_at: datetime,
    ) -> StudyAttempt:
        return self._services.study.record_answer(
            package_name,
            question_number,
            answer_text=answer_text,
            answered_at=answered_at,
        )

    def report(self, package_name: str, *, as_of: datetime) -> LearningProgressReport:
        return self._services.progress.report(package_name, as_of=as_of)

    @staticmethod
    def _build_services(library_directory: Path) -> _LibraryServices:
        database_path = library_directory / "metadata.sqlite3"
        documents = SqliteDocumentRepository(database_path)
        package_repository = SqliteLearningPackageRepository(database_path)
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
        return _LibraryServices(packages, summaries, questions, study, progress)


def _validate_library_name(name: str) -> str:
    normalized_name = name.strip()
    if not 1 <= len(normalized_name) <= 100:
        raise ValueError("Library name must contain between 1 and 100 characters")
    if any(ord(character) < 32 for character in normalized_name):
        raise ValueError("Library name must not contain control characters")
    return normalized_name
