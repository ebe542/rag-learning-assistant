from dataclasses import replace

import pytest

from rag_learning_assistant.generation.identity import (
    GenerationIdentity,
)
from rag_learning_assistant.generation.prompts import (
    PromptTemplate,
)


def test_generation_identity_is_stable_for_equal_configuration() -> None:
    prompt = PromptTemplate(
        name="summarization.map",
        version=1,
        text="Summarize supplied contexts.",
    )
    values = {
        "model_name": "Qwen/Qwen3-1.7B",
        "model_revision": "a" * 40,
        "prompt_references": (prompt.reference,),
        "max_map_new_tokens": 128,
        "max_reduce_new_tokens": 256,
        "max_batch_chars": 12_000,
        "document_content_sha256": "b" * 64,
    }

    first = GenerationIdentity(**values)
    second = GenerationIdentity(**values)

    assert first == second
    assert first.fingerprint == second.fingerprint
    assert len(first.fingerprint) == 64
    assert first.fingerprint == ("3f98bae5f6999621812171c7a097b62d394a4daa0de3150480b455089b71ff80")


@pytest.mark.parametrize(
    ("field", "changed_value"),
    [
        ("model_name", "Qwen/Qwen3-4B"),
        ("model_revision", "c" * 40),
        ("max_map_new_tokens", 96),
        ("max_reduce_new_tokens", 512),
        ("max_batch_chars", 24_000),
        ("document_content_sha256", "d" * 64),
    ],
)
def test_generation_identity_changes_with_configuration(
    field: str,
    changed_value: object,
) -> None:
    prompt = PromptTemplate(
        name="summarization.map",
        version=1,
        text="Summarize supplied contexts.",
    )
    identity = GenerationIdentity(
        model_name="Qwen/Qwen3-1.7B",
        model_revision="a" * 40,
        prompt_references=(prompt.reference,),
        max_map_new_tokens=128,
        max_reduce_new_tokens=256,
        max_batch_chars=12_000,
        document_content_sha256="b" * 64,
    )

    changed = replace(
        identity,
        **{field: changed_value},
    )

    assert changed.fingerprint != identity.fingerprint


def test_generation_identity_changes_with_prompt_reference() -> None:
    first_prompt = PromptTemplate(
        name="summarization.map",
        version=1,
        text="Summarize supplied contexts.",
    )
    second_prompt = PromptTemplate(
        name="summarization.map",
        version=2,
        text="Create a concise grounded summary.",
    )
    identity = GenerationIdentity(
        model_name="Qwen/Qwen3-1.7B",
        model_revision="a" * 40,
        prompt_references=(first_prompt.reference,),
        max_map_new_tokens=128,
        max_reduce_new_tokens=256,
        max_batch_chars=12_000,
        document_content_sha256="b" * 64,
    )

    changed = replace(
        identity,
        prompt_references=(second_prompt.reference,),
    )

    assert changed.fingerprint != identity.fingerprint


@pytest.mark.parametrize("field", ["model_name", "model_revision"])
def test_generation_identity_rejects_blank_model_fields(
    field: str,
) -> None:
    prompt = PromptTemplate(
        name="summarization.map",
        version=1,
        text="Summarize supplied contexts.",
    )
    identity = GenerationIdentity(
        model_name="Qwen/Qwen3-1.7B",
        model_revision="a" * 40,
        prompt_references=(prompt.reference,),
        max_map_new_tokens=128,
        max_reduce_new_tokens=256,
        max_batch_chars=12_000,
        document_content_sha256="b" * 64,
    )

    with pytest.raises(
        ValueError,
        match="Generation model fields must not be blank",
    ):
        replace(identity, **{field: "   "})


def test_generation_identity_requires_prompt_references() -> None:
    prompt = PromptTemplate(
        name="summarization.map",
        version=1,
        text="Summarize supplied contexts.",
    )
    identity = GenerationIdentity(
        model_name="Qwen/Qwen3-1.7B",
        model_revision="a" * 40,
        prompt_references=(prompt.reference,),
        max_map_new_tokens=128,
        max_reduce_new_tokens=256,
        max_batch_chars=12_000,
        document_content_sha256="b" * 64,
    )

    with pytest.raises(
        ValueError,
        match="Generation identity requires at least one prompt",
    ):
        replace(identity, prompt_references=())


@pytest.mark.parametrize(
    "field",
    [
        "max_map_new_tokens",
        "max_reduce_new_tokens",
        "max_batch_chars",
    ],
)
def test_generation_identity_requires_positive_limits(
    field: str,
) -> None:
    prompt = PromptTemplate(
        name="summarization.map",
        version=1,
        text="Summarize supplied contexts.",
    )
    identity = GenerationIdentity(
        model_name="Qwen/Qwen3-1.7B",
        model_revision="a" * 40,
        prompt_references=(prompt.reference,),
        max_map_new_tokens=128,
        max_reduce_new_tokens=256,
        max_batch_chars=12_000,
        document_content_sha256="b" * 64,
    )

    with pytest.raises(
        ValueError,
        match="Generation limits must be positive",
    ):
        replace(identity, **{field: 0})


@pytest.mark.parametrize(
    "content_hash",
    [
        "",
        "abc",
        "g" * 64,
        "A" * 64,
    ],
)
def test_generation_identity_requires_document_sha256(
    content_hash: str,
) -> None:
    prompt = PromptTemplate(
        name="summarization.map",
        version=1,
        text="Summarize supplied contexts.",
    )

    with pytest.raises(
        ValueError,
        match=("Document content hash must be a lowercase SHA-256 hex digest"),
    ):
        GenerationIdentity(
            model_name="Qwen/Qwen3-1.7B",
            model_revision="a" * 40,
            prompt_references=(prompt.reference,),
            max_map_new_tokens=128,
            max_reduce_new_tokens=256,
            max_batch_chars=12_000,
            document_content_sha256=content_hash,
        )


def test_generation_identity_rejects_duplicate_prompts() -> None:
    prompt = PromptTemplate(
        name="summarization.map",
        version=1,
        text="Summarize supplied contexts.",
    )

    with pytest.raises(
        ValueError,
        match="Generation prompts must be unique",
    ):
        GenerationIdentity(
            model_name="Qwen/Qwen3-1.7B",
            model_revision="a" * 40,
            prompt_references=(
                prompt.reference,
                prompt.reference,
            ),
            max_map_new_tokens=128,
            max_reduce_new_tokens=256,
            max_batch_chars=12_000,
            document_content_sha256="b" * 64,
        )


def test_generation_identity_is_part_of_public_generation_api() -> None:
    from rag_learning_assistant.generation import (
        GenerationIdentity as PublicGenerationIdentity,
    )

    assert PublicGenerationIdentity is GenerationIdentity
