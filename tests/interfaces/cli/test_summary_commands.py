import json
from pathlib import Path
from uuid import UUID

import pytest

from rag_learning_assistant.application import (
    DocumentNotFoundError,
    DocumentSummaryNotFoundError,
)
from rag_learning_assistant.chunking import TextChunker
from rag_learning_assistant.generation import (
    Citation,
    PersistedDocumentSummary,
    PromptReference,
    SqliteDocumentSummaryRepository,
)
from rag_learning_assistant.interfaces.cli import commands, entrypoint
from rag_learning_assistant.interfaces.cli.parser import build_parser
from rag_learning_assistant.library import SqliteDocumentRepository


class RecordingSummaryCatalog:
    def __init__(
        self,
        summaries: list[PersistedDocumentSummary],
    ) -> None:
        self.summaries = summaries
        self.document_ids: list[UUID] = []

    def list_document_summaries(
        self,
        document_id: UUID,
    ) -> list[PersistedDocumentSummary]:
        self.document_ids.append(document_id)
        return list(self.summaries)


class ShowingSummaryCatalog:
    def __init__(self, summary: PersistedDocumentSummary) -> None:
        self.summary = summary
        self.calls: list[tuple[UUID, str]] = []

    def get_document_summary(
        self,
        document_id: UUID,
        identity_fingerprint: str,
    ) -> PersistedDocumentSummary:
        self.calls.append((document_id, identity_fingerprint))
        return self.summary


def test_library_builder_configures_summary_lifecycle_repository(
    tmp_path: Path,
    monkeypatch,
) -> None:
    fake_indexer = object()

    def fake_build_persistent_document_search(
        chunker: TextChunker,
        index_directory: Path,
    ) -> object:
        return fake_indexer

    # Avoid loading embedding and FAISS dependencies in this wiring test.
    monkeypatch.setattr(
        commands,
        "build_persistent_document_search",
        fake_build_persistent_document_search,
    )

    service = commands.build_library_service(
        TextChunker(max_chars=1000, overlap_chars=100),
        tmp_path,
    )

    summary_cleaner = service.derived_data_cleaners[0]

    assert isinstance(
        summary_cleaner,
        SqliteDocumentSummaryRepository,
    )
    assert summary_cleaner.database_path == tmp_path / "metadata.sqlite3"


def test_parser_accepts_summary_list_command() -> None:
    args = build_parser().parse_args(
        [
            "summary-list",
            "local-data/indexes/library",
            "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa",
        ]
    )

    assert args.command == "summary-list"
    assert args.index_dir == Path("local-data/indexes/library")
    assert args.document_id == UUID(
        "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa",
    )


def test_summary_catalog_builder_uses_library_database(
    tmp_path: Path,
) -> None:
    catalog = commands.build_document_summary_catalog(tmp_path)

    assert isinstance(catalog.documents, SqliteDocumentRepository)
    assert catalog.documents.database_path == tmp_path / "metadata.sqlite3"
    assert isinstance(
        catalog.summaries,
        SqliteDocumentSummaryRepository,
    )
    assert catalog.summaries.database_path == tmp_path / "metadata.sqlite3"


def test_run_summary_list_outputs_version_metadata_as_json(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    document_id = UUID("aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa")
    summary = PersistedDocumentSummary(
        document_id=document_id,
        identity_fingerprint="b" * 64,
        source="document.pdf",
        text="Full persisted summary.",
        citations=(
            Citation(
                number=1,
                source="document.pdf",
                page_number=2,
                chunk_index=3,
                excerpt="Supporting passage.",
            ),
        ),
        prompt_references=(
            PromptReference(
                name="summarization.reduce",
                version=2,
                fingerprint="c" * 64,
            ),
        ),
    )
    catalog = RecordingSummaryCatalog([summary])
    monkeypatch.setattr(
        commands,
        "build_document_summary_catalog",
        lambda index_directory: catalog,
    )

    exit_code = commands.run_summary_list(tmp_path, document_id)

    assert exit_code == 0
    assert catalog.document_ids == [document_id]
    assert json.loads(capsys.readouterr().out) == {
        "index_directory": str(tmp_path),
        "document_id": str(document_id),
        "summaries": [
            {
                "identity_fingerprint": "b" * 64,
                "source": "document.pdf",
                "citation_count": 1,
                "prompts": [
                    {
                        "name": "summarization.reduce",
                        "version": 2,
                        "fingerprint": "c" * 64,
                    }
                ],
            }
        ],
    }


def test_entrypoint_dispatches_summary_list_command(
    monkeypatch,
) -> None:
    document_id = UUID("aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa")
    calls: list[tuple[Path, UUID]] = []

    def fake_run_summary_list(
        index_directory: Path,
        document_id: UUID,
    ) -> int:
        calls.append((index_directory, document_id))
        return 0

    monkeypatch.setattr(
        entrypoint.commands,
        "run_summary_list",
        fake_run_summary_list,
    )
    monkeypatch.setattr(
        entrypoint,
        "validate_library_directory",
        lambda path: None,
    )

    exit_code = entrypoint.main(
        [
            "summary-list",
            "local-data/indexes/library",
            str(document_id),
        ]
    )

    assert exit_code == 0
    assert calls == [
        (
            Path("local-data/indexes/library"),
            document_id,
        )
    ]


def test_entrypoint_reports_unknown_summary_document_as_cli_error(
    monkeypatch,
    capsys,
) -> None:
    document_id = UUID("aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa")

    def fail_with_unknown_document(
        index_directory: Path,
        document_id: UUID,
    ) -> int:
        raise DocumentNotFoundError(
            f"Document does not exist: {document_id}",
        )

    monkeypatch.setattr(
        entrypoint.commands,
        "run_summary_list",
        fail_with_unknown_document,
    )
    monkeypatch.setattr(
        entrypoint,
        "validate_library_directory",
        lambda path: None,
    )

    with pytest.raises(SystemExit):
        entrypoint.main(
            [
                "summary-list",
                "local-data/indexes/library",
                str(document_id),
            ]
        )

    assert f"Document does not exist: {document_id}" in capsys.readouterr().err


def test_parser_accepts_summary_show_command() -> None:
    document_id = UUID("aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa")
    identity_fingerprint = "b" * 64

    args = build_parser().parse_args(
        [
            "summary-show",
            "local-data/indexes/library",
            str(document_id),
            identity_fingerprint,
        ]
    )

    assert args.command == "summary-show"
    assert args.index_dir == Path("local-data/indexes/library")
    assert args.document_id == document_id
    assert args.identity_fingerprint == identity_fingerprint


@pytest.mark.parametrize(
    "identity_fingerprint",
    [
        "too-short",
        "g" * 64,
    ],
)
def test_parser_rejects_invalid_summary_fingerprint(
    identity_fingerprint: str,
) -> None:
    with pytest.raises(SystemExit):
        build_parser().parse_args(
            [
                "summary-show",
                "local-data/indexes/library",
                "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa",
                identity_fingerprint,
            ]
        )


def test_parser_normalizes_summary_fingerprint_to_lowercase() -> None:
    args = build_parser().parse_args(
        [
            "summary-show",
            "local-data/indexes/library",
            "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa",
            "A" * 64,
        ]
    )

    assert args.identity_fingerprint == "a" * 64


def test_run_summary_show_outputs_full_summary_as_json(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    document_id = UUID("aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa")
    identity_fingerprint = "b" * 64
    summary = PersistedDocumentSummary(
        document_id=document_id,
        identity_fingerprint=identity_fingerprint,
        source="document.pdf",
        text="Full persisted summary.",
        citations=(
            Citation(
                number=1,
                source="document.pdf",
                page_number=2,
                chunk_index=3,
                excerpt="Supporting passage.",
            ),
        ),
        prompt_references=(
            PromptReference(
                name="summarization.reduce",
                version=2,
                fingerprint="c" * 64,
            ),
        ),
    )
    catalog = ShowingSummaryCatalog(summary)
    monkeypatch.setattr(
        commands,
        "build_document_summary_catalog",
        lambda index_directory: catalog,
    )

    exit_code = commands.run_summary_show(
        tmp_path,
        document_id,
        identity_fingerprint,
    )

    assert exit_code == 0
    assert catalog.calls == [
        (document_id, identity_fingerprint),
    ]
    assert json.loads(capsys.readouterr().out) == {
        "index_directory": str(tmp_path),
        "document_id": str(document_id),
        "identity_fingerprint": identity_fingerprint,
        "source": "document.pdf",
        "summary": "Full persisted summary.",
        "citations": [
            {
                "number": 1,
                "source": "document.pdf",
                "page_number": 2,
                "chunk_index": 3,
                "excerpt": "Supporting passage.",
            }
        ],
        "prompts": [
            {
                "name": "summarization.reduce",
                "version": 2,
                "fingerprint": "c" * 64,
            }
        ],
    }


def test_entrypoint_dispatches_summary_show_command(
    monkeypatch,
) -> None:
    document_id = UUID("aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa")
    identity_fingerprint = "b" * 64
    calls: list[tuple[Path, UUID, str]] = []

    def fake_run_summary_show(
        index_directory: Path,
        document_id: UUID,
        identity_fingerprint: str,
    ) -> int:
        calls.append(
            (
                index_directory,
                document_id,
                identity_fingerprint,
            )
        )
        return 0

    monkeypatch.setattr(
        entrypoint.commands,
        "run_summary_show",
        fake_run_summary_show,
    )
    monkeypatch.setattr(
        entrypoint,
        "validate_library_directory",
        lambda path: None,
    )

    exit_code = entrypoint.main(
        [
            "summary-show",
            "local-data/indexes/library",
            str(document_id),
            identity_fingerprint,
        ]
    )

    assert exit_code == 0
    assert calls == [
        (
            Path("local-data/indexes/library"),
            document_id,
            identity_fingerprint,
        )
    ]


def test_entrypoint_reports_unknown_summary_identity_as_cli_error(
    monkeypatch,
    capsys,
) -> None:
    document_id = UUID("aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa")
    identity_fingerprint = "b" * 64

    def fail_with_unknown_summary(
        index_directory: Path,
        document_id: UUID,
        identity_fingerprint: str,
    ) -> int:
        raise DocumentSummaryNotFoundError(
            f"Stored document summary does not exist: {document_id}/{identity_fingerprint}"
        )

    monkeypatch.setattr(
        entrypoint.commands,
        "run_summary_show",
        fail_with_unknown_summary,
    )
    monkeypatch.setattr(
        entrypoint,
        "validate_library_directory",
        lambda path: None,
    )

    with pytest.raises(SystemExit):
        entrypoint.main(
            [
                "summary-show",
                "local-data/indexes/library",
                str(document_id),
                identity_fingerprint,
            ]
        )

    assert "Stored document summary does not exist" in capsys.readouterr().err
