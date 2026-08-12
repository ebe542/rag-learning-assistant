from dataclasses import replace
from pathlib import Path

import pytest

from rag_learning_assistant.generation import GenerationResult, PromptTemplate
from rag_learning_assistant.generation.cache import (
    CachedSummaryBatch,
)
from rag_learning_assistant.generation.sqlite_cache import (
    SqliteSummaryCache,
)


def test_cached_batch_survives_reopening(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "metadata.sqlite3"
    prompt = PromptTemplate(
        name="generation.system-json",
        version=1,
        text="Return valid JSON.",
    )

    batch = CachedSummaryBatch(
        identity_fingerprint="a" * 64,
        batch_number=1,
        first_context_number=1,
        last_context_number=3,
        result=GenerationResult(
            text="This section introduces Python.",
            citation_numbers=(1, 3),
            prompt_references=(prompt.reference,),
        ),
    )

    SqliteSummaryCache(database_path).save_batch(batch)

    reopened = SqliteSummaryCache(database_path)

    assert (
        reopened.find_batch(
            identity_fingerprint="a" * 64,
            batch_number=1,
        )
        == batch
    )


def test_find_batch_returns_none_for_unknown_entry(
    tmp_path: Path,
) -> None:
    cache = SqliteSummaryCache(tmp_path / "metadata.sqlite3")

    assert (
        cache.find_batch(
            identity_fingerprint="a" * 64,
            batch_number=1,
        )
        is None
    )


def test_saving_identical_batch_is_idempotent(
    tmp_path: Path,
) -> None:
    cache = SqliteSummaryCache(tmp_path / "metadata.sqlite3")
    batch = CachedSummaryBatch(
        identity_fingerprint="a" * 64,
        batch_number=1,
        first_context_number=1,
        last_context_number=2,
        result=GenerationResult(
            text="A cached section.",
            citation_numbers=(1,),
        ),
    )

    cache.save_batch(batch)
    cache.save_batch(batch)

    assert (
        cache.find_batch(
            identity_fingerprint="a" * 64,
            batch_number=1,
        )
        == batch
    )


def test_saving_conflicting_batch_is_rejected(
    tmp_path: Path,
) -> None:
    cache = SqliteSummaryCache(tmp_path / "metadata.sqlite3")
    batch = CachedSummaryBatch(
        identity_fingerprint="a" * 64,
        batch_number=1,
        first_context_number=1,
        last_context_number=2,
        result=GenerationResult(
            text="Original section.",
            citation_numbers=(1,),
        ),
    )
    cache.save_batch(batch)

    with pytest.raises(
        ValueError,
        match="Cached summary batch conflicts with existing data",
    ):
        cache.save_batch(
            replace(
                batch,
                result=GenerationResult(
                    text="Different section.",
                    citation_numbers=(2,),
                ),
            )
        )
