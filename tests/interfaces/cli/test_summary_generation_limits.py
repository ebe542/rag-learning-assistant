from pathlib import Path
from uuid import UUID

import pytest

from rag_learning_assistant.interfaces.cli import commands, entrypoint
from rag_learning_assistant.interfaces.cli.parser import (
    DEFAULT_SUMMARY_MAX_BATCH_CHARS,
    DEFAULT_SUMMARY_MAX_MAP_NEW_TOKENS,
    DEFAULT_SUMMARY_MAX_REDUCE_NEW_TOKENS,
    build_parser,
)


def test_parser_accepts_separate_summary_token_limits() -> None:
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


def test_parser_uses_separate_summary_token_defaults() -> None:
    args = build_parser().parse_args(
        [
            "summarize",
            "local-data/indexes/learning",
            "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa",
        ]
    )

    assert DEFAULT_SUMMARY_MAX_MAP_NEW_TOKENS == 192
    # The final reduction must have enough room for the document-wide answer
    # and the complete conservative citation union required by validation.
    assert DEFAULT_SUMMARY_MAX_REDUCE_NEW_TOKENS == 384

    assert args.max_map_new_tokens == DEFAULT_SUMMARY_MAX_MAP_NEW_TOKENS
    assert args.max_reduce_new_tokens == DEFAULT_SUMMARY_MAX_REDUCE_NEW_TOKENS
    assert args.max_batch_chars == DEFAULT_SUMMARY_MAX_BATCH_CHARS


def test_entrypoint_forwards_separate_summary_token_limits(
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
            "128",
            "--max-reduce-new-tokens",
            "256",
            "--max-batch-chars",
            "8000",
        ]
    )

    assert exit_code == 0
    assert calls == [
        (
            index_directory,
            document_id,
            128,
            256,
            8000,
            False,
        )
    ]
