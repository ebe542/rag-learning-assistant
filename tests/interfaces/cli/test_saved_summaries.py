from pathlib import Path
from uuid import UUID

from rag_learning_assistant.generation import (
    SqliteDocumentSummaryRepository,
)
from rag_learning_assistant.interfaces.cli import entrypoint
from rag_learning_assistant.interfaces.cli.commands import (
    build_document_summarization_service,
)
from rag_learning_assistant.interfaces.cli.parser import build_parser


def test_summarize_parser_accepts_force() -> None:
    args = build_parser().parse_args(
        [
            "summarize",
            "local-data/indexes/library",
            "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa",
            "--force",
        ]
    )

    assert args.force is True


def test_summarize_parser_does_not_force_by_default() -> None:
    args = build_parser().parse_args(
        [
            "summarize",
            "local-data/indexes/library",
            "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa",
        ]
    )

    assert args.force is False


def test_entrypoint_forwards_force_to_summarize_command(
    monkeypatch,
) -> None:
    calls: list[bool] = []

    def fake_run_summarize(
        index_directory: Path,
        document_id: UUID,
        max_map_new_tokens: int,
        max_reduce_new_tokens: int,
        max_batch_chars: int,
        force: bool,
    ) -> int:
        calls.append(force)
        return 0

    monkeypatch.setattr(entrypoint.commands, "run_summarize", fake_run_summarize)
    monkeypatch.setattr(
        entrypoint,
        "validate_existing_index_directory",
        lambda path: None,
    )

    exit_code = entrypoint.main(
        [
            "summarize",
            "local-data/indexes/library",
            "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa",
            "--force",
        ]
    )

    assert exit_code == 0
    assert calls == [True]


def test_summary_builder_configures_final_summary_repository(
    tmp_path: Path,
) -> None:
    service = build_document_summarization_service(
        index_directory=tmp_path,
        max_map_new_tokens=192,
        max_reduce_new_tokens=384,
        max_batch_chars=8000,
    )

    assert isinstance(
        service.final_summaries,
        SqliteDocumentSummaryRepository,
    )
    assert service.final_summaries.database_path == tmp_path / "metadata.sqlite3"
