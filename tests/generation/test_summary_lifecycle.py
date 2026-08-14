from uuid import UUID

from rag_learning_assistant.generation import (
    SqliteDocumentSummaryRepository,
)

from .test_summary_repository import build_summary


def test_delete_document_removes_all_its_summary_versions(tmp_path) -> None:
    repository = SqliteDocumentSummaryRepository(tmp_path / "metadata.sqlite3")
    document_id = UUID("aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa")
    other_document_id = UUID("bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb")

    first_version = build_summary(
        document_id=document_id,
        identity_fingerprint="1" * 64,
    )
    second_version = build_summary(
        document_id=document_id,
        identity_fingerprint="2" * 64,
    )
    other_summary = build_summary(
        document_id=other_document_id,
        identity_fingerprint="3" * 64,
    )

    repository.save(first_version)
    repository.save(second_version)
    repository.save(other_summary)

    deleted_count = repository.delete_document(document_id)

    assert deleted_count == 2
    assert repository.find(document_id, first_version.identity_fingerprint) is None
    assert repository.find(document_id, second_version.identity_fingerprint) is None
    assert (
        repository.find(
            other_document_id,
            other_summary.identity_fingerprint,
        )
        == other_summary
    )


def test_delete_unknown_document_is_idempotent(tmp_path) -> None:
    repository = SqliteDocumentSummaryRepository(tmp_path / "metadata.sqlite3")

    deleted_count = repository.delete_document(
        UUID("aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa"),
    )

    assert deleted_count == 0


def test_list_document_returns_only_its_summary_versions(tmp_path) -> None:
    repository = SqliteDocumentSummaryRepository(tmp_path / "metadata.sqlite3")
    document_id = UUID("aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa")
    other_document_id = UUID("bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb")

    second_version = build_summary(
        document_id=document_id,
        identity_fingerprint="2" * 64,
        text="Second version.",
    )
    first_version = build_summary(
        document_id=document_id,
        identity_fingerprint="1" * 64,
        text="First version.",
    )
    other_summary = build_summary(
        document_id=other_document_id,
        identity_fingerprint="3" * 64,
        text="Other document.",
    )

    repository.save(second_version)
    repository.save(first_version)
    repository.save(other_summary)

    summaries = repository.list_document(document_id)

    assert summaries == [first_version, second_version]


def test_list_unknown_document_returns_empty_list(tmp_path) -> None:
    repository = SqliteDocumentSummaryRepository(tmp_path / "metadata.sqlite3")

    summaries = repository.list_document(
        UUID("aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa"),
    )

    assert summaries == []
