from pathlib import Path

import pytest

from rag_learning_assistant.generation import (
    CachedQuestionBatch,
    GeneratedQuestionDraft,
    PromptReference,
    QuestionGenerationResult,
    SqliteQuestionBatchCache,
)


def build_result(
    *,
    first_question_number: int = 1,
    question_count: int = 1,
) -> QuestionGenerationResult:
    prompt_reference = PromptReference(
        name="question-bank",
        version=1,
        fingerprint="a" * 64,
    )
    return QuestionGenerationResult(
        questions=tuple(
            GeneratedQuestionDraft(
                number=number,
                text=f"Question {number}?",
                expected_answer=f"Answer {number}.",
                citation_numbers=(1,),
            )
            for number in range(
                first_question_number,
                first_question_number + question_count,
            )
        ),
        prompt_references=(prompt_reference,),
    )


def build_result_for_numbers(
    *question_numbers: int,
) -> QuestionGenerationResult:
    prompt_reference = PromptReference(
        name="question-bank",
        version=1,
        fingerprint="a" * 64,
    )
    return QuestionGenerationResult(
        questions=tuple(
            GeneratedQuestionDraft(
                number=number,
                text=f"Question {number}?",
                expected_answer=f"Answer {number}.",
                citation_numbers=(1,),
            )
            for number in question_numbers
        ),
        prompt_references=(prompt_reference,),
    )


def test_cached_question_batch_preserves_generated_questions() -> None:
    prompt_reference = PromptReference(
        name="question-bank",
        version=1,
        fingerprint="a" * 64,
    )
    result = QuestionGenerationResult(
        questions=(
            GeneratedQuestionDraft(
                number=6,
                text="What is an embedding?",
                expected_answer="An embedding is a numerical representation.",
                citation_numbers=(1,),
            ),
            GeneratedQuestionDraft(
                number=7,
                text="Why are embeddings useful?",
                expected_answer="They enable semantic comparisons.",
                citation_numbers=(2,),
            ),
        ),
        prompt_references=(prompt_reference,),
    )

    batch = CachedQuestionBatch(
        identity_fingerprint="b" * 64,
        batch_number=2,
        first_question_number=6,
        last_question_number=7,
        result=result,
    )

    assert batch.identity_fingerprint == "b" * 64
    assert batch.batch_number == 2
    assert batch.first_question_number == 6
    assert batch.last_question_number == 7
    assert batch.result == result


@pytest.mark.parametrize(
    "identity_fingerprint",
    [
        "",
        "not-a-sha256",
        "A" * 64,
        "a" * 63,
    ],
)
def test_cached_question_batch_requires_sha256_identity(
    identity_fingerprint: str,
) -> None:
    with pytest.raises(
        ValueError,
        match="Question batch identity must be a lowercase SHA-256 hex digest",
    ):
        CachedQuestionBatch(
            identity_fingerprint=identity_fingerprint,
            batch_number=1,
            first_question_number=1,
            last_question_number=1,
            result=build_result(),
        )


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        (
            "batch_number",
            0,
            "Question batch number must be positive",
        ),
        (
            "batch_number",
            -1,
            "Question batch number must be positive",
        ),
        (
            "first_question_number",
            0,
            "First question number must be positive",
        ),
        (
            "first_question_number",
            -1,
            "First question number must be positive",
        ),
        (
            "last_question_number",
            0,
            "Last question number must be positive",
        ),
        (
            "last_question_number",
            -1,
            "Last question number must be positive",
        ),
    ],
)
def test_cached_question_batch_requires_positive_numbers(
    field: str,
    value: int,
    message: str,
) -> None:
    arguments = {
        "identity_fingerprint": "b" * 64,
        "batch_number": 1,
        "first_question_number": 1,
        "last_question_number": 1,
        "result": build_result(),
    }
    arguments[field] = value

    with pytest.raises(ValueError, match=message):
        CachedQuestionBatch(**arguments)


def test_cached_question_batch_rejects_reversed_question_range() -> None:
    with pytest.raises(
        ValueError,
        match="Last question number must not precede first question number",
    ):
        CachedQuestionBatch(
            identity_fingerprint="b" * 64,
            batch_number=2,
            first_question_number=6,
            last_question_number=5,
            result=build_result(),
        )


@pytest.mark.parametrize(
    ("first_question_number", "last_question_number", "result"),
    [
        (
            6,
            7,
            build_result_for_numbers(1, 2),
        ),
        (
            6,
            7,
            build_result_for_numbers(6),
        ),
        (
            6,
            7,
            build_result_for_numbers(6, 8),
        ),
        (
            6,
            8,
            build_result_for_numbers(6, 8, 7),
        ),
    ],
)
def test_cached_question_batch_requires_exact_contiguous_question_range(
    first_question_number: int,
    last_question_number: int,
    result: QuestionGenerationResult,
) -> None:
    with pytest.raises(
        ValueError,
        match="Generated question numbers must match the cached batch range",
    ):
        CachedQuestionBatch(
            identity_fingerprint="b" * 64,
            batch_number=2,
            first_question_number=first_question_number,
            last_question_number=last_question_number,
            result=result,
        )


def test_question_batch_survives_reopening(tmp_path: Path) -> None:
    database_path = tmp_path / "metadata.sqlite3"
    batch = CachedQuestionBatch(
        identity_fingerprint="b" * 64,
        batch_number=2,
        first_question_number=6,
        last_question_number=7,
        result=build_result_for_numbers(6, 7),
    )

    cache = SqliteQuestionBatchCache(database_path)
    cache.save_batch(batch)

    reopened_cache = SqliteQuestionBatchCache(database_path)

    assert (
        reopened_cache.find_batch(
            identity_fingerprint="b" * 64,
            batch_number=2,
        )
        == batch
    )


def test_saving_identical_question_batch_is_idempotent(
    tmp_path: Path,
) -> None:
    cache = SqliteQuestionBatchCache(tmp_path / "metadata.sqlite3")
    batch = CachedQuestionBatch(
        identity_fingerprint="b" * 64,
        batch_number=1,
        first_question_number=1,
        last_question_number=1,
        result=build_result_for_numbers(1),
    )

    cache.save_batch(batch)
    cache.save_batch(batch)

    assert (
        cache.find_batch(
            identity_fingerprint="b" * 64,
            batch_number=1,
        )
        == batch
    )


def test_question_batch_cache_rejects_conflicting_content(
    tmp_path: Path,
) -> None:
    cache = SqliteQuestionBatchCache(tmp_path / "metadata.sqlite3")
    original = CachedQuestionBatch(
        identity_fingerprint="b" * 64,
        batch_number=1,
        first_question_number=1,
        last_question_number=1,
        result=build_result_for_numbers(1),
    )
    conflicting = CachedQuestionBatch(
        identity_fingerprint="b" * 64,
        batch_number=1,
        first_question_number=1,
        last_question_number=1,
        result=QuestionGenerationResult(
            questions=(
                GeneratedQuestionDraft(
                    number=1,
                    text="A different question?",
                    expected_answer="A different answer.",
                    citation_numbers=(1,),
                ),
            ),
            prompt_references=original.result.prompt_references,
        ),
    )

    cache.save_batch(original)

    with pytest.raises(
        ValueError,
        match="Cached question batch conflicts with existing data",
    ):
        cache.save_batch(conflicting)

    assert (
        cache.find_batch(
            identity_fingerprint="b" * 64,
            batch_number=1,
        )
        == original
    )


def test_question_batch_cache_returns_none_for_missing_batch(
    tmp_path: Path,
) -> None:
    cache = SqliteQuestionBatchCache(tmp_path / "metadata.sqlite3")

    assert (
        cache.find_batch(
            identity_fingerprint="b" * 64,
            batch_number=1,
        )
        is None
    )
