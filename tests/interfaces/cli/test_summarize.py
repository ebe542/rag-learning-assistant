import json
from pathlib import Path
from types import SimpleNamespace
from uuid import UUID

import pytest

from rag_learning_assistant.application import (
    DocumentNotFoundError,
    DocumentSummary,
)
from rag_learning_assistant.generation import (
    Citation,
    GenerationResult,
    PromptTemplate,
    SqliteSummaryCache,
)
from rag_learning_assistant.interfaces.cli import commands, entrypoint
from rag_learning_assistant.interfaces.cli.parser import (
    DEFAULT_SUMMARY_MAX_BATCH_CHARS,
    DEFAULT_SUMMARY_MAX_MAP_NEW_TOKENS,
    DEFAULT_SUMMARY_MAX_REDUCE_NEW_TOKENS,
    build_parser,
)
from rag_learning_assistant.library import IndexedDocument


class RecordingSummarizationService:
    def __init__(self, summary: DocumentSummary) -> None:
        self.summary = summary
        self.requested_ids: list[UUID] = []

    def summarize(
        self,
        document_id: UUID,
        *,
        force: bool = False,
    ) -> DocumentSummary:
        self.requested_ids.append(document_id)
        return self.summary


class StubGenerator:
    """Expose stable model metadata without loading a real local model."""

    model_name = "example/summary-model"
    model_revision = "revision-1"

    def __init__(self, max_new_tokens: int) -> None:
        self.max_new_tokens = max_new_tokens

    def generate(
        self,
        prompt: str,
        *,
        max_new_tokens: int | None = None,
    ) -> GenerationResult:
        raise AssertionError(f"Generation is not expected while wiring the service: {prompt}")


def test_summary_builder_configures_persistent_cache_identity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    diagnostics: list[tuple[str, str, dict[str, object]]] = []
    monkeypatch.setattr(
        commands,
        "SentenceTransformerEmbedder",
        lambda: SimpleNamespace(model_name="example/embedder", model_revision="revision-2"),
    )
    monkeypatch.setattr(commands, "FaissVectorStore", lambda *args, **kwargs: object())
    monkeypatch.setattr(
        commands,
        "SqliteDocumentRepository",
        lambda database_path: object(),
    )
    monkeypatch.setattr(commands, "HuggingFaceTextGenerator", StubGenerator)
    monkeypatch.setattr(
        commands,
        "write_diagnostic_log",
        lambda message, *, source, context: diagnostics.append((message, source, context)),
    )

    service = commands.build_document_summarization_service(
        index_directory=tmp_path,
        max_map_new_tokens=160,
        max_reduce_new_tokens=320,
        max_batch_chars=36_000,
    )

    assert isinstance(service.cache, SqliteSummaryCache)
    assert service.cache.database_path == tmp_path / "metadata.sqlite3"
    assert service.identity_factory is not None

    document = IndexedDocument(
        id=UUID("aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa"),
        source="course.pdf",
        content_sha256="a" * 64,
        page_count=10,
        chunk_count=20,
    )
    identity = service.identity_factory(document)

    assert identity.model_name == StubGenerator.model_name
    assert identity.model_revision == StubGenerator.model_revision
    assert identity.max_map_new_tokens == 160
    assert identity.max_reduce_new_tokens == 320
    assert identity.max_batch_chars == 36_000
    assert identity.document_content_sha256 == document.content_sha256
    assert {reference.name for reference in identity.prompt_references} == {
        "generation.json-repair",
        "generation.system-json",
        "summarization.map",
        "summarization.reduce",
        "summarization.reduce-coverage-repair",
    }
    assert service.warning is not None

    service.warning("Fallback used")

    assert diagnostics == [
        (
            "Fallback used",
            "summarization",
            {"index_directory": tmp_path},
        )
    ]


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
    calls: list[tuple[Path, UUID, int, int, int, bool]] = []

    def fake_run_summarize(
        index_directory: Path,
        document_id: UUID,
        max_map_new_tokens: int,
        max_reduce_new_tokens: int,
        max_batch_chars: int,
        force: bool,
    ) -> int:
        calls.append(
            (
                index_directory,
                document_id,
                max_map_new_tokens,
                max_reduce_new_tokens,
                max_batch_chars,
                force,
            )
        )
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
            "--max-map-new-tokens",
            "160",
            "--max-reduce-new-tokens",
            "320",
            "--max-batch-chars",
            "36000",
        ]
    )

    assert exit_code == 0
    assert calls == [
        (
            index_directory,
            document_id,
            160,
            320,
            36_000,
            False,
        )
    ]


def test_run_summarize_outputs_grounded_summary_as_json(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    document_id = UUID("aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa")
    prompt = PromptTemplate(
        name="summarization.map",
        version=1,
        text="Summarize supplied contexts.",
    )
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
            prompt_references=(prompt.reference,),
        )
    )

    builder_calls: list[tuple[Path, int, int, int]] = []

    def fake_build_document_summarization_service(
        index_directory: Path,
        max_map_new_tokens: int,
        max_reduce_new_tokens: int,
        max_batch_chars: int,
    ) -> RecordingSummarizationService:
        builder_calls.append(
            (
                index_directory,
                max_map_new_tokens,
                max_reduce_new_tokens,
                max_batch_chars,
            )
        )
        return service

    monkeypatch.setattr(
        commands,
        "build_document_summarization_service",
        fake_build_document_summarization_service,
    )

    exit_code = commands.run_summarize(
        index_directory=tmp_path,
        document_id=document_id,
        max_map_new_tokens=160,
        max_reduce_new_tokens=320,
        max_batch_chars=36_000,
    )

    assert exit_code == 0
    assert builder_calls == [(tmp_path, 160, 320, 36_000)]
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
        "prompts": [
            {
                "name": prompt.name,
                "version": prompt.version,
                "fingerprint": prompt.fingerprint,
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
        max_map_new_tokens: int,
        max_reduce_new_tokens: int,
        max_batch_chars: int,
        force: bool,
    ) -> int:
        assert index_directory == tmp_path / "library"
        assert max_map_new_tokens == DEFAULT_SUMMARY_MAX_MAP_NEW_TOKENS
        assert max_reduce_new_tokens == DEFAULT_SUMMARY_MAX_REDUCE_NEW_TOKENS
        assert max_batch_chars == DEFAULT_SUMMARY_MAX_BATCH_CHARS

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


def test_parser_accepts_summary_token_limits() -> None:
    args = build_parser().parse_args(
        [
            "summarize",
            "local-data/indexes/learning",
            "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa",
            "--max-map-new-tokens",
            "128",
            "--max-reduce-new-tokens",
            "256",
        ]
    )

    assert args.max_map_new_tokens == 128
    assert args.max_reduce_new_tokens == 256


def test_parser_uses_default_summary_token_limits() -> None:
    args = build_parser().parse_args(
        [
            "summarize",
            "local-data/indexes/learning",
            "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa",
        ]
    )

    assert args.max_map_new_tokens == DEFAULT_SUMMARY_MAX_MAP_NEW_TOKENS
    assert args.max_reduce_new_tokens == DEFAULT_SUMMARY_MAX_REDUCE_NEW_TOKENS


@pytest.mark.parametrize(
    "option",
    ["--max-map-new-tokens", "--max-reduce-new-tokens"],
)
@pytest.mark.parametrize("token_limit", ["0", "-1"])
def test_parser_rejects_non_positive_summary_token_limits(
    option: str,
    token_limit: str,
) -> None:
    with pytest.raises(SystemExit):
        build_parser().parse_args(
            [
                "summarize",
                "local-data/indexes/learning",
                "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa",
                option,
                token_limit,
            ]
        )


def test_summary_progress_is_written_to_stderr(
    capsys: pytest.CaptureFixture[str],
) -> None:
    commands.write_summarization_progress("map", 2, 7)
    commands.write_summarization_progress("reduce", 1, 1)

    captured = capsys.readouterr()

    assert captured.out == ""
    assert captured.err == ("Summarizing batch 2/7...\nCombining partial summaries...\n")


def test_parser_accepts_summary_batch_size() -> None:
    args = build_parser().parse_args(
        [
            "summarize",
            "local-data/indexes/learning",
            "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa",
            "--max-batch-chars",
            "36000",
        ]
    )

    assert args.max_batch_chars == 36_000


def test_parser_uses_default_summary_batch_size() -> None:
    args = build_parser().parse_args(
        [
            "summarize",
            "local-data/indexes/learning",
            "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa",
        ]
    )

    assert args.max_batch_chars == DEFAULT_SUMMARY_MAX_BATCH_CHARS


@pytest.mark.parametrize("max_batch_chars", ["0", "-1"])
def test_parser_rejects_non_positive_summary_batch_size(
    max_batch_chars: str,
) -> None:
    with pytest.raises(SystemExit):
        build_parser().parse_args(
            [
                "summarize",
                "local-data/indexes/learning",
                "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa",
                "--max-batch-chars",
                max_batch_chars,
            ]
        )
