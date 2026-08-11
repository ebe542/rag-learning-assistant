from hashlib import sha256

import pytest

from rag_learning_assistant.generation.prompts import (
    PromptReference,
    PromptTemplate,
)


def test_prompt_template_exposes_versioned_text_and_fingerprint() -> None:
    text = "Answer only from the provided sources."
    prompt = PromptTemplate(
        name="question-answering.grounded-answer",
        version=1,
        text=text,
    )

    assert prompt.name == "question-answering.grounded-answer"
    assert prompt.version == 1
    assert prompt.text == text
    assert prompt.fingerprint == sha256(text.encode("utf-8")).hexdigest()


@pytest.mark.parametrize("name", ["", "   "])
def test_prompt_template_rejects_blank_name(name: str) -> None:
    with pytest.raises(
        ValueError,
        match="Prompt name must not be blank",
    ):
        PromptTemplate(
            name=name,
            version=1,
            text="Valid prompt text.",
        )


@pytest.mark.parametrize("version", [0, -1])
def test_prompt_template_requires_positive_version(
    version: int,
) -> None:
    with pytest.raises(
        ValueError,
        match="Prompt version must be positive",
    ):
        PromptTemplate(
            name="test.prompt",
            version=version,
            text="Valid prompt text.",
        )


@pytest.mark.parametrize("text", ["", "   "])
def test_prompt_template_rejects_blank_text(text: str) -> None:
    with pytest.raises(
        ValueError,
        match="Prompt text must not be blank",
    ):
        PromptTemplate(
            name="test.prompt",
            version=1,
            text=text,
        )


def test_prompt_template_creates_compact_reference() -> None:
    prompt = PromptTemplate(
        name="summarization.map",
        version=1,
        text="Summarize only the supplied contexts.",
    )

    reference = prompt.reference

    assert reference == PromptReference(
        name="summarization.map",
        version=1,
        fingerprint=prompt.fingerprint,
    )


@pytest.mark.parametrize("name", ["", "   "])
def test_prompt_reference_rejects_blank_name(name: str) -> None:
    with pytest.raises(
        ValueError,
        match="Prompt reference name must not be blank",
    ):
        PromptReference(
            name=name,
            version=1,
            fingerprint="a" * 64,
        )


@pytest.mark.parametrize("version", [0, -1])
def test_prompt_reference_requires_positive_version(
    version: int,
) -> None:
    with pytest.raises(
        ValueError,
        match="Prompt reference version must be positive",
    ):
        PromptReference(
            name="test.prompt",
            version=version,
            fingerprint="a" * 64,
        )


@pytest.mark.parametrize(
    "fingerprint",
    [
        "",
        "abc",
        "g" * 64,
        "A" * 64,
    ],
)
def test_prompt_reference_requires_sha256_fingerprint(
    fingerprint: str,
) -> None:
    with pytest.raises(
        ValueError,
        match="Prompt fingerprint must be a lowercase SHA-256 hex digest",
    ):
        PromptReference(
            name="test.prompt",
            version=1,
            fingerprint=fingerprint,
        )


def test_prompt_types_are_part_of_public_generation_api() -> None:
    from rag_learning_assistant.generation import (
        PromptReference as PublicPromptReference,
    )
    from rag_learning_assistant.generation import (
        PromptTemplate as PublicPromptTemplate,
    )

    assert PublicPromptReference is PromptReference
    assert PublicPromptTemplate is PromptTemplate
