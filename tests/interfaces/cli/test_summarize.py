import json
from pathlib import Path
from uuid import UUID

import pytest

from rag_learning_assistant.application import (
    DocumentNotFoundError,
    DocumentSummary,
)
from rag_learning_assistant.generation import Citation
from rag_learning_assistant.interfaces.cli import commands, entrypoint
from rag_learning_assistant.interfaces.cli.parser import build_parser


class RecordingSummarizationService:
    def __init__(self, summary: DocumentSummary) -> None:
        self.summary = summary
        self.requested_ids: list[UUID] = []

    def summarize(self, document_id: UUID) -> DocumentSummary:
        self.requested_ids.append(document_id)
        return self.summary


def test_parser_accepts_summarize_command() -> None:
    document_id = UUID("aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa")

    args = build_parser().parse_args(
        [
            "summarize",
            "local-data/indexes/learning",
            str(document_id),
        ]
    )

    assert args.command == "summarize"
    assert args.index_dir == Path("local-data/indexes/learning")
    assert args.document_id == document_id


def test_entrypoint_dispatches_summarize_command(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    index_directory = tmp_path / "library"
    index_directory.mkdir()
    (index_directory / "vectors.faiss").touch()
    (index_directory / "metadata.sqlite3").touch()

    document_id = UUID("aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa")
    calls: list[tuple[Path, UUID]] = []

    def fake_run_summarize(
        index_directory: Path,
        document_id: UUID,
    ) -> int:
        calls.append((index_directory, document_id))
        return 0

    # raising=False allows this red test to exist before run_summarize does.
    monkeypatch.setattr(
        commands,
        "run_summarize",
        fake_run_summarize,
    )

    exit_code = entrypoint.main(
        [
            "summarize",
            str(index_directory),
            str(document_id),
        ]
    )

    assert exit_code == 0
    assert calls == [(index_directory, document_id)]


def test_run_summarize_outputs_grounded_summary_as_json(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    document_id = UUID("aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa")
    service = RecordingSummarizationService(
        DocumentSummary(
            document_id=document_id,
            source="course.pdf",
            text="The document introduces Python fundamentals.",
            citations=(
                Citation(
                    number=2,
                    source="course.pdf",
                    page_number=4,
                    chunk_index=7,
                    excerpt="Python programs consist of instructions.",
                ),
            ),
        )
    )

    monkeypatch.setattr(
        commands,
        "build_document_summarization_service",
        lambda index_directory: service,
        raising=False,
    )

    exit_code = commands.run_summarize(
        index_directory=tmp_path,
        document_id=document_id,
    )

    assert exit_code == 0
    assert service.requested_ids == [document_id]
    assert json.loads(capsys.readouterr().out) == {
        "document_id": str(document_id),
        "source": "course.pdf",
        "summary": "The document introduces Python fundamentals.",
        "citations": [
            {
                "number": 2,
                "source": "course.pdf",
                "page_number": 4,
                "chunk_index": 7,
                "excerpt": "Python programs consist of instructions.",
            }
        ],
    }


def test_entrypoint_rejects_incomplete_summarize_index_before_loading_service(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    index_directory = tmp_path / "incomplete-library"
    index_directory.mkdir()
    (index_directory / "metadata.sqlite3").touch()

    def fail_if_called(**kwargs: object) -> int:
        raise AssertionError("Summarization must not start for an incomplete index")

    monkeypatch.setattr(
        commands,
        "run_summarize",
        fail_if_called,
    )

    with pytest.raises(SystemExit) as exc_info:
        entrypoint.main(
            [
                "summarize",
                str(index_directory),
                "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa",
            ]
        )

    assert exc_info.value.code == 2
    assert "index directory is incomplete" in capsys.readouterr().err


def test_entrypoint_reports_unknown_document_as_cli_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    index_directory = tmp_path / "library"
    index_directory.mkdir()
    (index_directory / "vectors.faiss").touch()
    (index_directory / "metadata.sqlite3").touch()

    document_id = UUID("aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa")

    def fail_with_unknown_document(
        index_directory: Path,
        document_id: UUID,
    ) -> int:
        raise DocumentNotFoundError(f"Document does not exist: {document_id}")

    monkeypatch.setattr(
        commands,
        "run_summarize",
        fail_with_unknown_document,
    )

    with pytest.raises(SystemExit) as exc_info:
        entrypoint.main(
            [
                "summarize",
                str(index_directory),
                str(document_id),
            ]
        )

    assert exc_info.value.code == 2
    assert f"Document does not exist: {document_id}" in capsys.readouterr().err
