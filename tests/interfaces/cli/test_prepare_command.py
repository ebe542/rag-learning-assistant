import json
from pathlib import Path
from uuid import UUID

import pytest

from rag_learning_assistant.interfaces.cli import commands, entrypoint
from rag_learning_assistant.interfaces.cli.parser import (
    DEFAULT_QUESTION_COUNT,
    build_parser,
)
from rag_learning_assistant.learning import (
    LearningPackage,
    LearningPackageStatus,
    SqliteLearningPackageRepository,
)


def test_parser_accepts_user_facing_prepare_command() -> None:
    args = build_parser().parse_args(
        [
            "prepare",
            "books/python-basics.pdf",
            "--library",
            "local-data/library",
        ]
    )

    assert args.command == "prepare"
    assert args.pdf == Path("books/python-basics.pdf")
    assert args.library == Path("local-data/library")
    assert args.name is None
    assert args.question_count == DEFAULT_QUESTION_COUNT


def test_prepare_parser_accepts_product_options() -> None:
    args = build_parser().parse_args(
        [
            "prepare",
            "books/python-basics.pdf",
            "--library",
            "local-data/library",
            "--name",
            "Python Basics",
            "--questions",
            "25",
        ]
    )

    assert args.name == "Python Basics"
    assert args.question_count == 25


@pytest.mark.parametrize(
    "arguments",
    [
        ["prepare", "books/python-basics.pdf"],
        ["package-list"],
        ["package-show", "--package", "Python Basics"],
        ["package-remove", "--package", "Python Basics"],
        ["progress", "--package", "Python Basics"],
    ],
)
def test_product_commands_use_configured_default_library(
    arguments: list[str],
    tmp_path: Path,
    monkeypatch,
) -> None:
    library_directory = tmp_path / "personal-library"
    monkeypatch.setenv("RAG_LEARN_LIBRARY", str(library_directory))

    args = build_parser().parse_args(arguments)

    assert args.library == library_directory


def test_entrypoint_dispatches_prepare_with_default_name(
    tmp_path: Path,
    monkeypatch,
) -> None:
    library_directory = tmp_path / "library"
    calls: list[tuple[Path, Path, str, int]] = []

    def fake_run_prepare(
        pdf_path: Path,
        library_directory: Path,
        name: str,
        question_count: int,
    ) -> int:
        calls.append(
            (
                pdf_path,
                library_directory,
                name,
                question_count,
            )
        )
        return 0

    monkeypatch.setattr(
        commands,
        "run_prepare",
        fake_run_prepare,
        raising=False,
    )

    result = entrypoint.main(
        [
            "prepare",
            "books/python-basics.pdf",
            "--library",
            str(library_directory),
        ]
    )

    assert result == 0
    assert calls == [
        (
            Path("books/python-basics.pdf"),
            library_directory,
            "python-basics",
            DEFAULT_QUESTION_COUNT,
        )
    ]


class RecordingLearningPackageService:
    def __init__(self, result: LearningPackage) -> None:
        self.result = result
        self.calls: list[tuple[str, Path, int]] = []
        self.remove_calls: list[str] = []

    def prepare(
        self,
        *,
        name: str,
        pdf_path: Path,
        question_count: int,
    ) -> LearningPackage:
        self.calls.append((name, pdf_path, question_count))
        return self.result

    def remove(self, name: str) -> LearningPackage:
        self.remove_calls.append(name)
        return self.result


def test_run_prepare_outputs_ready_package_as_json(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    library_directory = tmp_path / "library"
    package = LearningPackage(
        id=UUID("aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa"),
        name="Python Basics",
        document_id=UUID("bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb"),
        status=LearningPackageStatus.READY,
        summary_identity_fingerprint="c" * 64,
        question_bank_identity_fingerprint="d" * 64,
    )
    service = RecordingLearningPackageService(package)

    monkeypatch.setattr(
        commands,
        "build_learning_package_service",
        lambda directory: service,
        raising=False,
    )

    result = commands.run_prepare(
        pdf_path=Path("books/python-basics.pdf"),
        library_directory=library_directory,
        name="Python Basics",
        question_count=20,
    )

    assert result == 0
    assert service.calls == [
        (
            "Python Basics",
            Path("books/python-basics.pdf"),
            20,
        )
    ]
    assert json.loads(capsys.readouterr().out) == {
        "library_directory": str(library_directory),
        "package": {
            "id": "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa",
            "name": "Python Basics",
            "document_id": ("bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb"),
            "status": "ready",
            "summary_identity_fingerprint": "c" * 64,
            "question_bank_identity_fingerprint": "d" * 64,
        },
    }


class StubDocumentImporter:
    def add_document(self, path: Path):
        raise AssertionError("Builder test must not import a document")


class StubSummaryPreparer:
    def prepare_summary(self, document_id: UUID) -> str:
        raise AssertionError("Builder test must not summarize")


class StubQuestionPreparer:
    def prepare_questions(
        self,
        document_id: UUID,
        summary_identity_fingerprint: str,
        *,
        question_count: int,
    ) -> str:
        raise AssertionError("Builder test must not generate questions")


def test_learning_package_builder_composes_existing_services(
    tmp_path: Path,
    monkeypatch,
) -> None:
    documents = StubDocumentImporter()
    summaries = StubSummaryPreparer()
    questions = StubQuestionPreparer()
    calls: list[tuple[str, Path]] = []

    def fake_build_library_service(
        chunker,
        index_directory: Path,
    ) -> StubDocumentImporter:
        calls.append(("documents", index_directory))
        return documents

    def fake_build_summary_service(
        index_directory: Path,
        max_map_new_tokens: int,
        max_reduce_new_tokens: int,
        max_batch_chars: int,
    ) -> StubSummaryPreparer:
        calls.append(("summaries", index_directory))
        return summaries

    def fake_build_question_service(
        index_directory: Path,
        max_new_tokens: int,
    ) -> StubQuestionPreparer:
        calls.append(("questions", index_directory))
        return questions

    monkeypatch.setattr(
        commands,
        "build_library_service",
        fake_build_library_service,
    )
    monkeypatch.setattr(
        commands,
        "build_document_summarization_service",
        fake_build_summary_service,
    )
    monkeypatch.setattr(
        commands,
        "build_question_bank_service",
        fake_build_question_service,
    )

    service = commands.build_learning_package_service(tmp_path)

    assert service.documents is documents
    assert service.summaries is summaries
    assert service.questions is questions
    assert isinstance(
        service.packages,
        SqliteLearningPackageRepository,
    )
    assert calls == [
        ("documents", tmp_path),
        ("summaries", tmp_path),
        ("questions", tmp_path),
    ]


@pytest.mark.parametrize(
    ("phase", "expected"),
    [
        ("index", "Indexing document..."),
        ("summarize", "Creating document summary..."),
        ("questions", "Generating study questions..."),
        ("ready", "Learning package is ready."),
    ],
)
def test_learning_package_progress_is_human_readable(
    phase: str,
    expected: str,
    capsys,
) -> None:
    commands.write_learning_package_progress(phase)

    captured = capsys.readouterr()

    assert captured.out == ""
    assert captured.err.strip() == expected


def test_parser_accepts_package_list_command() -> None:
    args = build_parser().parse_args(
        [
            "package-list",
            "--library",
            "local-data/library",
        ]
    )

    assert args.command == "package-list"
    assert args.library == Path("local-data/library")


class StaticLearningPackageCatalog:
    def __init__(
        self,
        packages: list[LearningPackage],
    ) -> None:
        self.packages = packages

    def list_packages(self) -> list[LearningPackage]:
        return list(self.packages)

    def get_package(self, name: str) -> LearningPackage:
        return next(package for package in self.packages if package.name == name)


def test_run_package_list_outputs_available_packages(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    library_directory = tmp_path / "library"
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
            status=LearningPackageStatus.READY,
            summary_identity_fingerprint="e" * 64,
            question_bank_identity_fingerprint="f" * 64,
        ),
    ]
    monkeypatch.setattr(
        commands,
        "build_learning_package_catalog",
        lambda directory: StaticLearningPackageCatalog(packages),
        raising=False,
    )

    result = commands.run_package_list(library_directory)

    assert result == 0
    assert json.loads(capsys.readouterr().out) == {
        "library_directory": str(library_directory),
        "packages": [
            {
                "id": "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa",
                "name": "Algorithms",
                "document_id": ("bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb"),
                "status": "indexed",
                "summary_identity_fingerprint": None,
                "question_bank_identity_fingerprint": None,
            },
            {
                "id": "cccccccc-cccc-cccc-cccc-cccccccccccc",
                "name": "Python Basics",
                "document_id": ("dddddddd-dddd-dddd-dddd-dddddddddddd"),
                "status": "ready",
                "summary_identity_fingerprint": "e" * 64,
                "question_bank_identity_fingerprint": "f" * 64,
            },
        ],
    }


def test_entrypoint_dispatches_package_list(
    tmp_path: Path,
    monkeypatch,
) -> None:
    library_directory = tmp_path / "library"
    library_directory.mkdir()
    (library_directory / "metadata.sqlite3").write_bytes(b"")
    calls: list[Path] = []

    def fake_run_package_list(
        directory: Path,
    ) -> int:
        calls.append(directory)
        return 0

    monkeypatch.setattr(
        commands,
        "run_package_list",
        fake_run_package_list,
    )

    result = entrypoint.main(
        [
            "package-list",
            "--library",
            str(library_directory),
        ]
    )

    assert result == 0
    assert calls == [library_directory]


@pytest.mark.parametrize("command", ["package-show", "package-remove"])
def test_parser_accepts_named_package_commands(command: str) -> None:
    args = build_parser().parse_args(
        [
            command,
            "--library",
            "local-data/library",
            "--package",
            "Python Basics",
        ]
    )

    assert args.command == command
    assert args.library == Path("local-data/library")
    assert args.package == "Python Basics"


def test_run_package_show_outputs_selected_package(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    package = LearningPackage(
        id=UUID("aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa"),
        name="Python Basics",
        document_id=UUID("bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb"),
        status=LearningPackageStatus.INDEXED,
    )
    monkeypatch.setattr(
        commands,
        "build_learning_package_catalog",
        lambda directory: StaticLearningPackageCatalog([package]),
    )

    assert commands.run_package_show(tmp_path, "Python Basics") == 0
    assert json.loads(capsys.readouterr().out)["package"]["name"] == "Python Basics"


def test_run_package_remove_uses_product_service(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    package = LearningPackage(
        id=UUID("aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa"),
        name="Python Basics",
        document_id=UUID("bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb"),
        status=LearningPackageStatus.INDEXED,
    )
    service = RecordingLearningPackageService(package)
    monkeypatch.setattr(
        commands,
        "build_learning_package_service",
        lambda directory: service,
    )

    assert commands.run_package_remove(tmp_path, "Python Basics") == 0
    assert service.remove_calls == ["Python Basics"]
    assert json.loads(capsys.readouterr().out)["removed_package"]["name"] == "Python Basics"
