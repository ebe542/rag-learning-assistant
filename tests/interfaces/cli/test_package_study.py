from datetime import UTC, datetime, timedelta
from pathlib import Path
from uuid import UUID

import pytest

from rag_learning_assistant.application import (
    DueQuestion,
    LearningPackageStudyService,
    StudySessionService,
)
from rag_learning_assistant.generation import Citation
from rag_learning_assistant.interfaces.cli import commands, entrypoint
from rag_learning_assistant.interfaces.cli.parser import build_parser
from rag_learning_assistant.learning import (
    AnswerEvaluation,
    AnswerVerdict,
    QuestionProgress,
    ReviewRating,
    SqliteLearningPackageRepository,
    StudyAttempt,
    StudyQuestion,
)

AS_OF = datetime(2026, 8, 20, 12, 0, tzinfo=UTC)


class RecordingPackageStudyService:
    def __init__(
        self,
        due: DueQuestion | None = None,
        attempt: StudyAttempt | None = None,
    ) -> None:
        self.due = due
        self.attempt = attempt
        self.calls: list[tuple[str, datetime]] = []
        self.record_calls: list[tuple[str, int, str, datetime]] = []

    def next_due(
        self,
        package_name: str,
        *,
        as_of: datetime,
    ) -> DueQuestion | None:
        self.calls.append((package_name, as_of))
        return self.due

    def record_answer(
        self,
        package_name: str,
        question_number: int,
        *,
        answer_text: str,
        answered_at: datetime,
    ) -> StudyAttempt:
        self.record_calls.append(
            (
                package_name,
                question_number,
                answer_text,
                answered_at,
            )
        )

        if self.attempt is None:
            raise AssertionError("No study attempt configured")

        return self.attempt


def test_parser_accepts_study_by_learning_package_name() -> None:
    args = build_parser().parse_args(
        [
            "study",
            "--library",
            "product-library",
            "--package",
            "RAG Learning Assistant",
        ]
    )

    assert args.command == "study"
    assert args.library == Path("product-library")
    assert args.package == "RAG Learning Assistant"
    assert args.index_dir is None
    assert args.document_id is None
    assert args.question_bank_identity_fingerprint is None


def test_entrypoint_uses_default_library_for_package_study(
    tmp_path: Path,
    monkeypatch,
) -> None:
    (tmp_path / "metadata.sqlite3").write_bytes(b"")
    calls: list[tuple[Path, str]] = []
    monkeypatch.setenv("RAG_LEARN_LIBRARY", str(tmp_path))
    monkeypatch.setattr(
        commands,
        "run_package_study",
        lambda library_directory, package_name: (
            calls.append((library_directory, package_name)) or 0
        ),
    )

    exit_code = entrypoint.main(
        [
            "study",
            "--package",
            "RAG Learning Assistant",
        ]
    )

    assert exit_code == 0
    assert calls == [(tmp_path, "RAG Learning Assistant")]


def test_entrypoint_dispatches_study_by_package_name(
    tmp_path: Path,
    monkeypatch,
) -> None:
    (tmp_path / "metadata.sqlite3").write_bytes(b"")
    calls: list[tuple[Path, str]] = []

    def fake_run_package_study(
        library_directory: Path,
        package_name: str,
    ) -> int:
        calls.append((library_directory, package_name))
        return 0

    monkeypatch.setattr(
        commands,
        "run_package_study",
        fake_run_package_study,
    )

    exit_code = entrypoint.main(
        [
            "study",
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


def test_package_study_builder_uses_persistent_library_storage(
    tmp_path: Path,
) -> None:
    service = commands.build_learning_package_study_service(tmp_path)

    assert isinstance(service, LearningPackageStudyService)
    assert isinstance(
        service.packages,
        SqliteLearningPackageRepository,
    )
    assert isinstance(
        service.sessions,
        StudySessionService,
    )
    assert service.packages.database_path == (tmp_path / "metadata.sqlite3")


def test_run_package_study_reports_when_no_question_is_due(
    tmp_path: Path,
    monkeypatch,
) -> None:
    service = RecordingPackageStudyService()
    output: list[str] = []

    def fake_build_package_study_service(
        library_directory: Path,
    ) -> RecordingPackageStudyService:
        assert library_directory == tmp_path
        return service

    monkeypatch.setattr(
        commands,
        "build_learning_package_study_service",
        fake_build_package_study_service,
    )

    exit_code = commands.run_package_study(
        library_directory=tmp_path,
        package_name="RAG Learning Assistant",
        as_of=AS_OF,
        write_line=output.append,
    )

    assert exit_code == 0
    assert service.calls == [
        (
            "RAG Learning Assistant",
            AS_OF,
        )
    ]
    assert output == ["No study questions are due."]


def test_run_package_study_records_written_answer(
    tmp_path: Path,
    monkeypatch,
) -> None:
    question = StudyQuestion(
        number=1,
        text="What is retrieval?",
        expected_answer="Retrieval finds relevant source passages.",
        citations=(
            Citation(
                number=1,
                source="document.pdf",
                page_number=3,
                chunk_index=4,
                excerpt="Retrieval finds relevant source passages.",
            ),
        ),
    )
    progress = QuestionProgress(
        document_id=UUID("aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa"),
        question_bank_identity_fingerprint="b" * 64,
        question_number=1,
        repetition_count=1,
        interval_days=1,
        ease_factor=2.5,
        due_at=AS_OF + timedelta(days=1),
        last_reviewed_at=AS_OF,
    )
    evaluation = AnswerEvaluation(
        verdict=AnswerVerdict.PARTIALLY_CORRECT,
        score=0.7,
        feedback="Retrieval was identified.",
        missing_concepts=("Retrieval happens before generation.",),
    )
    attempt = StudyAttempt(
        id=UUID("dddddddd-dddd-dddd-dddd-dddddddddddd"),
        document_id=progress.document_id,
        question_bank_identity_fingerprint=(progress.question_bank_identity_fingerprint),
        question_number=1,
        question_text=question.text,
        answer_text="It finds relevant passages.",
        expected_answer=question.expected_answer,
        citations=question.citations,
        rating=ReviewRating.HARD,
        answered_at=AS_OF,
        resulting_progress=progress,
        evaluation=evaluation,
    )
    service = RecordingPackageStudyService(
        due=DueQuestion(
            question=question,
            progress=None,
        ),
        attempt=attempt,
    )
    inputs = iter(["It finds relevant passages."])
    output: list[str] = []

    monkeypatch.setattr(
        commands,
        "build_learning_package_study_service",
        lambda library_directory: service,
    )

    exit_code = commands.run_package_study(
        library_directory=tmp_path,
        package_name="RAG Learning Assistant",
        as_of=AS_OF,
        read_line=lambda prompt: next(inputs),
        write_line=output.append,
    )

    assert exit_code == 0
    assert service.record_calls == [
        (
            "RAG Learning Assistant",
            1,
            "It finds relevant passages.",
            AS_OF,
        )
    ]
    assert "Expected answer: Retrieval finds relevant source passages." in output
    assert "Evaluation: partially_correct (score: 0.70)" in output
    assert "Feedback: Retrieval was identified." in output
    assert "Missing concept: Retrieval happens before generation." in output
    assert "Scheduled as: hard" in output
    assert output[-1] == (f"Review recorded. Next due: {progress.due_at.isoformat()}")


@pytest.mark.parametrize(
    "arguments",
    [
        ["study", "--library", "product-library"],
    ],
)
def test_entrypoint_rejects_incomplete_package_selection(
    arguments: list[str],
    capsys,
) -> None:
    with pytest.raises(SystemExit):
        entrypoint.main(arguments)

    assert "--library requires --package" in capsys.readouterr().err


def test_entrypoint_rejects_mixed_study_selection(
    capsys,
) -> None:
    with pytest.raises(SystemExit):
        entrypoint.main(
            [
                "study",
                "technical-library",
                "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa",
                "b" * 64,
                "--library",
                "product-library",
                "--package",
                "RAG Learning Assistant",
            ]
        )

    assert "Package and technical study arguments must not be mixed" in capsys.readouterr().err
