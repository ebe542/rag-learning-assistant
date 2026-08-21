import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from rag_learning_assistant.application import (
    LearningPackageNotFoundError,
    LearningPackageNotReadyError,
    LearningProgressReport,
    LearningProgressService,
    QuestionBankCatalog,
)
from rag_learning_assistant.interfaces.cli import (
    commands,
    entrypoint,
)
from rag_learning_assistant.interfaces.cli.parser import build_parser
from rag_learning_assistant.learning import (
    SqliteLearningPackageRepository,
    SqliteQuestionProgressRepository,
    SqliteStudyAttemptRepository,
)

AS_OF = datetime(2026, 8, 21, 12, 0, tzinfo=UTC)


def test_parser_accepts_learning_progress_command() -> None:
    args = build_parser().parse_args(
        [
            "progress",
            "--library",
            "product-library",
            "--package",
            "RAG Learning Assistant",
        ]
    )

    assert args.command == "progress"
    assert args.library == Path("product-library")
    assert args.package == "RAG Learning Assistant"


def test_learning_progress_builder_uses_persistent_library_storage(
    tmp_path: Path,
) -> None:
    service = commands.build_learning_progress_service(tmp_path)

    assert isinstance(service, LearningProgressService)
    assert isinstance(
        service.packages,
        SqliteLearningPackageRepository,
    )
    assert isinstance(service.banks, QuestionBankCatalog)
    assert isinstance(
        service.progress,
        SqliteQuestionProgressRepository,
    )
    assert isinstance(
        service.attempts,
        SqliteStudyAttemptRepository,
    )

    database_path = tmp_path / "metadata.sqlite3"
    assert service.packages.database_path == database_path
    assert service.progress.database_path == database_path
    assert service.attempts.database_path == database_path


class RecordingLearningProgressService:
    def __init__(
        self,
        report: LearningProgressReport,
    ) -> None:
        self.progress_report = report
        self.calls: list[tuple[str, datetime]] = []

    def report(
        self,
        package_name: str,
        *,
        as_of: datetime,
    ) -> LearningProgressReport:
        self.calls.append((package_name, as_of))
        return self.progress_report


def test_run_progress_outputs_machine_readable_report(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    report = LearningProgressReport(
        package_name="RAG Learning Assistant",
        total_question_count=10,
        answered_question_count=4,
        due_question_count=3,
        attempt_count=6,
        incorrect_attempt_count=2,
        partially_correct_attempt_count=1,
        correct_attempt_count=3,
        difficult_concepts=(
            ("document identity", 2),
            ("citation relationships", 1),
        ),
        last_studied_at=AS_OF,
        next_due_at=AS_OF,
        unclassified_attempt_count=0,
    )
    service = RecordingLearningProgressService(report)

    monkeypatch.setattr(
        commands,
        "build_learning_progress_service",
        lambda library_directory: service,
    )

    exit_code = commands.run_progress(
        library_directory=tmp_path,
        package_name="RAG Learning Assistant",
        as_of=AS_OF,
    )

    assert exit_code == 0
    assert service.calls == [
        (
            "RAG Learning Assistant",
            AS_OF,
        )
    ]
    assert json.loads(capsys.readouterr().out) == {
        "library_directory": str(tmp_path),
        "package": "RAG Learning Assistant",
        "questions": {
            "total": 10,
            "answered": 4,
            "due": 3,
            "answered_rate": 0.4,
        },
        "attempts": {
            "total": 6,
            "incorrect": 2,
            "partially_correct": 1,
            "correct": 3,
            "unclassified": 0,
            "correct_rate": 0.5,
        },
        "difficult_concepts": [
            {
                "concept": "document identity",
                "missing_count": 2,
            },
            {
                "concept": "citation relationships",
                "missing_count": 1,
            },
        ],
        "last_studied_at": AS_OF.isoformat(),
        "next_due_at": AS_OF.isoformat(),
    }


def test_entrypoint_dispatches_learning_progress_command(
    tmp_path: Path,
    monkeypatch,
) -> None:
    (tmp_path / "metadata.sqlite3").write_bytes(b"")
    calls: list[tuple[Path, str]] = []

    def fake_run_progress(
        library_directory: Path,
        package_name: str,
    ) -> int:
        calls.append(
            (
                library_directory,
                package_name,
            )
        )
        return 0

    monkeypatch.setattr(
        commands,
        "run_progress",
        fake_run_progress,
    )

    exit_code = entrypoint.main(
        [
            "progress",
            "--library",
            str(tmp_path),
            "--package",
            "RAG Learning Assistant",
        ]
    )

    assert exit_code == 0
    assert calls == [
        (
            tmp_path,
            "RAG Learning Assistant",
        )
    ]


@pytest.mark.parametrize(
    "error",
    [
        LearningPackageNotFoundError("Learning package does not exist: Unknown"),
        LearningPackageNotReadyError("Learning package is not ready: RAG Learning Assistant"),
    ],
)
def test_entrypoint_reports_learning_progress_errors_as_cli_errors(
    error: LookupError | ValueError,
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    (tmp_path / "metadata.sqlite3").write_bytes(b"")

    def fail_progress(
        library_directory: Path,
        package_name: str,
    ) -> int:
        raise error

    monkeypatch.setattr(
        commands,
        "run_progress",
        fail_progress,
    )

    with pytest.raises(SystemExit):
        entrypoint.main(
            [
                "progress",
                "--library",
                str(tmp_path),
                "--package",
                "RAG Learning Assistant",
            ]
        )

    assert str(error) in capsys.readouterr().err
