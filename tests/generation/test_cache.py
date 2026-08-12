from dataclasses import replace

import pytest

from rag_learning_assistant.generation import (
    GenerationResult,
    PromptTemplate,
)
from rag_learning_assistant.generation.cache import (
    CachedSummaryBatch,
)


def test_cached_summary_batch_preserves_generated_result() -> None:
    prompt = PromptTemplate(
        name="generation.system-json",
        version=1,
        text="Return valid JSON.",
    )
    result = GenerationResult(
        text="This section introduces Python.",
        citation_numbers=(1, 3),
        prompt_references=(prompt.reference,),
    )

    cached = CachedSummaryBatch(
        identity_fingerprint="a" * 64,
        batch_number=1,
        first_context_number=1,
        last_context_number=3,
        result=result,
    )

    assert cached.identity_fingerprint == "a" * 64
    assert cached.batch_number == 1
    assert cached.first_context_number == 1
    assert cached.last_context_number == 3
    assert cached.result == result


def _cached_batch() -> CachedSummaryBatch:
    return CachedSummaryBatch(
        identity_fingerprint="a" * 64,
        batch_number=1,
        first_context_number=1,
        last_context_number=3,
        result=GenerationResult(
            text="A grounded section summary.",
            citation_numbers=(1, 3),
        ),
    )


@pytest.mark.parametrize(
    "fingerprint",
    ["", "abc", "g" * 64, "A" * 64],
)
def test_cached_batch_requires_generation_fingerprint(
    fingerprint: str,
) -> None:
    with pytest.raises(
        ValueError,
        match=("Generation fingerprint must be a lowercase SHA-256 hex digest"),
    ):
        replace(
            _cached_batch(),
            identity_fingerprint=fingerprint,
        )


@pytest.mark.parametrize("batch_number", [0, -1])
def test_cached_batch_requires_positive_batch_number(
    batch_number: int,
) -> None:
    with pytest.raises(
        ValueError,
        match="Batch number must be positive",
    ):
        replace(
            _cached_batch(),
            batch_number=batch_number,
        )


@pytest.mark.parametrize("context_number", [0, -1])
def test_cached_batch_requires_positive_context_numbers(
    context_number: int,
) -> None:
    with pytest.raises(
        ValueError,
        match="Context numbers must be positive",
    ):
        replace(
            _cached_batch(),
            first_context_number=context_number,
        )


def test_cached_batch_rejects_reversed_context_range() -> None:
    with pytest.raises(
        ValueError,
        match="Last context number must not precede first",
    ):
        replace(
            _cached_batch(),
            first_context_number=3,
            last_context_number=2,
        )


def test_cached_batch_rejects_citation_outside_context_range() -> None:
    with pytest.raises(
        ValueError,
        match="Cached citation does not belong to its batch",
    ):
        replace(
            _cached_batch(),
            result=GenerationResult(
                text="Invalid cached summary.",
                citation_numbers=(4,),
            ),
        )
