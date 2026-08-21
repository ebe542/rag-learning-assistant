from dataclasses import replace
from uuid import UUID

import pytest

from rag_learning_assistant.generation import Citation, PromptReference
from rag_learning_assistant.learning import (
    QuestionBank,
    QuestionBankIdentity,
    StudyQuestion,
)


def test_study_question_preserves_grounded_learning_content() -> None:
    citations = (
        Citation(
            number=1,
            source="course.pdf",
            page_number=4,
            chunk_index=7,
            excerpt="Embeddings represent text as numeric vectors.",
        ),
        Citation(
            number=2,
            source="course.pdf",
            page_number=5,
            chunk_index=9,
            excerpt="Similar meanings should have similar vectors.",
        ),
    )

    question = StudyQuestion(
        number=1,
        text="What is the purpose of a text embedding?",
        expected_answer=(
            "It represents text as a numeric vector so that semantic similarity can be compared."
        ),
        citations=citations,
    )

    assert question.number == 1
    assert question.text == "What is the purpose of a text embedding?"
    assert question.expected_answer.startswith(
        "It represents text as a numeric vector",
    )
    assert question.citations == citations


def build_question(**overrides: object) -> StudyQuestion:
    values = {
        "number": 1,
        "text": "What is a text embedding?",
        "expected_answer": "A numeric representation of text.",
        "citations": (
            Citation(
                number=1,
                source="course.pdf",
                page_number=4,
                chunk_index=7,
                excerpt="Embeddings represent text as numeric vectors.",
            ),
        ),
    }
    values.update(overrides)
    return StudyQuestion(**values)


@pytest.mark.parametrize("number", [0, -1])
def test_study_question_requires_positive_number(number: int) -> None:
    with pytest.raises(
        ValueError,
        match="Question number must be positive",
    ):
        build_question(number=number)


@pytest.mark.parametrize(
    ("field", "message"),
    [
        ("text", "Question text must not be blank"),
        (
            "expected_answer",
            "Question expected_answer must not be blank",
        ),
    ],
)
@pytest.mark.parametrize("value", ["", "   "])
def test_study_question_rejects_blank_text_fields(
    field: str,
    message: str,
    value: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        build_question(**{field: value})


def test_study_question_requires_at_least_one_citation() -> None:
    with pytest.raises(
        ValueError,
        match="Question requires at least one citation",
    ):
        build_question(citations=())


def test_study_question_rejects_duplicate_citation_numbers() -> None:
    citation = Citation(
        number=1,
        source="course.pdf",
        page_number=4,
        chunk_index=7,
        excerpt="Supporting passage.",
    )

    with pytest.raises(
        ValueError,
        match="Question citation numbers must be unique",
    ):
        build_question(citations=(citation, citation))


def test_question_bank_preserves_document_and_generation_identity() -> None:
    question = build_question()
    prompt = PromptReference(
        name="question-bank.generate",
        version=1,
        fingerprint="a" * 64,
    )

    bank = QuestionBank(
        document_id=UUID("aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa"),
        identity_fingerprint="b" * 64,
        source="course.pdf",
        questions=(question,),
        prompt_references=(prompt,),
    )

    assert bank.document_id == UUID(
        "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa",
    )
    assert bank.identity_fingerprint == "b" * 64
    assert bank.source == "course.pdf"
    assert bank.questions == (question,)
    assert bank.prompt_references == (prompt,)


def build_bank(**overrides: object) -> QuestionBank:
    values = {
        "document_id": UUID(
            "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa",
        ),
        "identity_fingerprint": "b" * 64,
        "source": "course.pdf",
        "questions": (build_question(),),
        "prompt_references": (
            PromptReference(
                name="question-bank.generate",
                version=1,
                fingerprint="a" * 64,
            ),
        ),
    }
    values.update(overrides)
    return QuestionBank(**values)


@pytest.mark.parametrize(
    ("field", "message"),
    [
        (
            "identity_fingerprint",
            "Question bank identity_fingerprint must not be blank",
        ),
        ("source", "Question bank source must not be blank"),
    ],
)
@pytest.mark.parametrize("value", ["", "   "])
def test_question_bank_rejects_blank_text_fields(
    field: str,
    message: str,
    value: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        build_bank(**{field: value})


def test_question_bank_requires_at_least_one_question() -> None:
    with pytest.raises(
        ValueError,
        match="Question bank requires at least one question",
    ):
        build_bank(questions=())


def test_question_bank_requires_prompt_reference() -> None:
    with pytest.raises(
        ValueError,
        match="Question bank requires at least one prompt reference",
    ):
        build_bank(prompt_references=())


def test_question_numbers_must_be_consecutive() -> None:
    with pytest.raises(
        ValueError,
        match="Question numbers must be consecutive starting at 1",
    ):
        build_bank(
            questions=(
                build_question(number=1),
                build_question(number=3),
            )
        )


def test_question_citations_must_belong_to_bank_source() -> None:
    foreign_citation = Citation(
        number=1,
        source="other.pdf",
        page_number=1,
        chunk_index=0,
        excerpt="Foreign supporting passage.",
    )

    with pytest.raises(
        ValueError,
        match="Question citation source must match question bank source",
    ):
        build_bank(questions=(build_question(citations=(foreign_citation,)),))


def test_question_bank_rejects_duplicate_question_texts() -> None:
    with pytest.raises(
        ValueError,
        match="Question bank question texts must be unique",
    ):
        build_bank(
            questions=(
                build_question(
                    number=1,
                    text="What is an embedding?",
                ),
                build_question(
                    number=2,
                    text="  WHAT IS AN EMBEDDING?  ",
                ),
            )
        )


def test_question_bank_identity_is_stable_for_equal_configuration() -> None:
    prompt = PromptReference(
        name="question-bank.generate",
        version=1,
        fingerprint="a" * 64,
    )
    values = {
        "model_name": "Qwen/Qwen3-1.7B",
        "model_revision": "b" * 40,
        "prompt_references": (prompt,),
        "question_count": 5,
        "batch_size": 5,
        "max_new_tokens": 512,
        "summary_identity_fingerprint": "c" * 64,
    }

    first = QuestionBankIdentity(**values)
    second = QuestionBankIdentity(**values)

    assert first == second
    assert first.fingerprint == second.fingerprint
    assert len(first.fingerprint) == 64


@pytest.mark.parametrize(
    ("field", "changed_value"),
    [
        ("model_name", "Qwen/Qwen3-4B"),
        ("model_revision", "d" * 40),
        (
            "prompt_references",
            (
                PromptReference(
                    name="question-bank.generate",
                    version=2,
                    fingerprint="e" * 64,
                ),
            ),
        ),
        ("question_count", 10),
        ("batch_size", 10),
        ("max_new_tokens", 768),
        ("summary_identity_fingerprint", "f" * 64),
    ],
)
def test_question_bank_identity_changes_with_configuration(
    field: str,
    changed_value: object,
) -> None:
    identity = QuestionBankIdentity(
        model_name="Qwen/Qwen3-1.7B",
        model_revision="b" * 40,
        prompt_references=(
            PromptReference(
                name="question-bank.generate",
                version=1,
                fingerprint="a" * 64,
            ),
        ),
        question_count=5,
        batch_size=5,
        max_new_tokens=512,
        summary_identity_fingerprint="c" * 64,
    )

    changed = replace(
        identity,
        **{field: changed_value},
    )

    assert changed.fingerprint != identity.fingerprint


def build_question_bank_identity(
    **overrides: object,
) -> QuestionBankIdentity:
    values = {
        "model_name": "Qwen/Qwen3-1.7B",
        "model_revision": "b" * 40,
        "prompt_references": (
            PromptReference(
                name="question-bank.generate",
                version=1,
                fingerprint="a" * 64,
            ),
        ),
        "question_count": 5,
        "batch_size": 5,
        "max_new_tokens": 512,
        "summary_identity_fingerprint": "c" * 64,
    }
    values.update(overrides)
    return QuestionBankIdentity(**values)


@pytest.mark.parametrize(
    "field",
    ["model_name", "model_revision"],
)
@pytest.mark.parametrize("value", ["", "   "])
def test_question_bank_identity_rejects_blank_model_fields(
    field: str,
    value: str,
) -> None:
    with pytest.raises(
        ValueError,
        match="Question bank model fields must not be blank",
    ):
        build_question_bank_identity(**{field: value})


def test_question_bank_identity_requires_prompt_reference() -> None:
    with pytest.raises(
        ValueError,
        match="Question bank identity requires at least one prompt",
    ):
        build_question_bank_identity(prompt_references=())


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("question_count", 0),
        ("question_count", -1),
        ("max_new_tokens", 0),
        ("max_new_tokens", -1),
        ("batch_size", 0),
        ("batch_size", -1),
    ],
)
def test_question_bank_identity_requires_positive_limits(
    field: str,
    value: int,
) -> None:
    with pytest.raises(
        ValueError,
        match="Question bank generation limits must be positive",
    ):
        build_question_bank_identity(**{field: value})


@pytest.mark.parametrize(
    "fingerprint",
    [
        "",
        "c" * 63,
        "c" * 65,
        "G" * 64,
    ],
)
def test_question_bank_identity_requires_summary_sha256(
    fingerprint: str,
) -> None:
    with pytest.raises(
        ValueError,
        match=("Summary identity fingerprint must be a lowercase SHA-256 hex digest"),
    ):
        build_question_bank_identity(
            summary_identity_fingerprint=fingerprint,
        )


def test_question_bank_identity_rejects_duplicate_prompts() -> None:
    prompt = PromptReference(
        name="question-bank.generate",
        version=1,
        fingerprint="a" * 64,
    )

    with pytest.raises(
        ValueError,
        match="Question bank prompts must be unique",
    ):
        build_question_bank_identity(
            prompt_references=(prompt, prompt),
        )


def test_question_bank_rejects_duplicate_prompt_references() -> None:
    prompt = PromptReference(
        name="question-bank.generate",
        version=1,
        fingerprint="a" * 64,
    )

    with pytest.raises(
        ValueError,
        match="Question bank prompts must be unique",
    ):
        build_bank(
            prompt_references=(prompt, prompt),
        )
