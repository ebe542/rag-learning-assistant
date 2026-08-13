from uuid import UUID

import pytest

from rag_learning_assistant.generation import (
    Citation,
    PersistedDocumentSummary,
    PromptReference,
    SqliteDocumentSummaryRepository,
)


def test_persisted_document_summary_preserves_generation_metadata() -> None:
    prompt = PromptReference(
        name="summarization.reduce",
        version=4,
        fingerprint="a" * 64,
    )

    summary = PersistedDocumentSummary(
        document_id=UUID("aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa"),
        identity_fingerprint="b" * 64,
        source="document.pdf",
        text="A grounded document summary.",
        citations=(
            Citation(
                number=1,
                source="document.pdf",
                page_number=1,
                chunk_index=0,
                excerpt="First supporting passage.",
            ),
            Citation(
                number=2,
                source="document.pdf",
                page_number=2,
                chunk_index=3,
                excerpt="Second supporting passage.",
            ),
        ),
        prompt_references=(prompt,),
    )

    assert summary.document_id == UUID("aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa")
    assert summary.identity_fingerprint == "b" * 64
    assert summary.source == "document.pdf"
    assert summary.text == "A grounded document summary."
    assert tuple(citation.number for citation in summary.citations) == (1, 2)
    assert summary.prompt_references == (prompt,)


def build_summary(**overrides: object) -> PersistedDocumentSummary:
    values = {
        "document_id": UUID("aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa"),
        "identity_fingerprint": "b" * 64,
        "source": "document.pdf",
        "text": "A grounded document summary.",
        "citations": (
            Citation(
                number=1,
                source="document.pdf",
                page_number=1,
                chunk_index=0,
                excerpt="First supporting passage.",
            ),
            Citation(
                number=2,
                source="document.pdf",
                page_number=2,
                chunk_index=3,
                excerpt="Second supporting passage.",
            ),
        ),
        "prompt_references": (
            PromptReference(
                name="summarization.reduce",
                version=4,
                fingerprint="a" * 64,
            ),
        ),
    }
    values.update(overrides)
    return PersistedDocumentSummary(**values)


@pytest.mark.parametrize("field", ["identity_fingerprint", "source", "text"])
@pytest.mark.parametrize("value", ["", "   "])
def test_persisted_summary_rejects_blank_text_fields(
    field: str,
    value: str,
) -> None:
    with pytest.raises(
        ValueError,
        match=f"Persisted summary {field} must not be blank",
    ):
        build_summary(**{field: value})


def test_persisted_summary_requires_citations() -> None:
    with pytest.raises(
        ValueError,
        match="Persisted summary requires at least one citation",
    ):
        build_summary(citations=())


def test_persisted_summary_rejects_duplicate_citation_numbers() -> None:
    citation = Citation(
        number=1,
        source="document.pdf",
        page_number=1,
        chunk_index=0,
        excerpt="Supporting passage.",
    )

    with pytest.raises(
        ValueError,
        match="Persisted summary citation numbers must be unique",
    ):
        build_summary(citations=(citation, citation))


def test_persisted_summary_requires_prompt_reference() -> None:
    with pytest.raises(
        ValueError,
        match="Persisted summary requires at least one prompt reference",
    ):
        build_summary(prompt_references=())


def test_persisted_summary_survives_repository_reopening(
    tmp_path,
) -> None:
    database_path = tmp_path / "metadata.sqlite3"
    summary = build_summary()

    repository = SqliteDocumentSummaryRepository(database_path)
    repository.save(summary)

    reopened_repository = SqliteDocumentSummaryRepository(database_path)

    assert (
        reopened_repository.find(
            summary.document_id,
            summary.identity_fingerprint,
        )
        == summary
    )


def test_repository_returns_none_for_unknown_summary(tmp_path) -> None:
    repository = SqliteDocumentSummaryRepository(tmp_path / "metadata.sqlite3")

    assert (
        repository.find(
            UUID("aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa"),
            "b" * 64,
        )
        is None
    )


def test_saving_identical_summary_is_idempotent(tmp_path) -> None:
    repository = SqliteDocumentSummaryRepository(tmp_path / "metadata.sqlite3")
    summary = build_summary()

    repository.save(summary)
    repository.save(summary)

    assert (
        repository.find(
            summary.document_id,
            summary.identity_fingerprint,
        )
        == summary
    )


def test_saving_conflicting_summary_is_rejected(tmp_path) -> None:
    repository = SqliteDocumentSummaryRepository(tmp_path / "metadata.sqlite3")
    original = build_summary()
    conflicting = build_summary(text="A different summary.")

    repository.save(original)

    with pytest.raises(
        ValueError,
        match="Conflicting final summary already exists",
    ):
        repository.save(conflicting)

    assert (
        repository.find(
            original.document_id,
            original.identity_fingerprint,
        )
        == original
    )


def test_replace_updates_existing_summary_explicitly(tmp_path) -> None:
    repository = SqliteDocumentSummaryRepository(tmp_path / "metadata.sqlite3")
    original = build_summary()
    replacement = build_summary(text="Explicitly regenerated summary.")

    repository.save(original)
    repository.replace(replacement)

    assert (
        repository.find(
            original.document_id,
            original.identity_fingerprint,
        )
        == replacement
    )
