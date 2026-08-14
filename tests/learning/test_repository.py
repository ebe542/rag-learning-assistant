from uuid import UUID

import pytest

from rag_learning_assistant.learning import (
    SqliteQuestionBankRepository,
)

from .test_models import build_bank, build_question


def test_question_bank_survives_repository_reopening(tmp_path) -> None:
    database_path = tmp_path / "metadata.sqlite3"
    bank = build_bank()

    repository = SqliteQuestionBankRepository(database_path)
    repository.save(bank)

    reopened_repository = SqliteQuestionBankRepository(database_path)

    assert (
        reopened_repository.find(
            bank.document_id,
            bank.identity_fingerprint,
        )
        == bank
    )


def test_repository_returns_none_for_unknown_question_bank(
    tmp_path,
) -> None:
    repository = SqliteQuestionBankRepository(
        tmp_path / "metadata.sqlite3",
    )

    assert (
        repository.find(
            UUID("aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa"),
            "b" * 64,
        )
        is None
    )


def test_saving_identical_question_bank_is_idempotent(
    tmp_path,
) -> None:
    repository = SqliteQuestionBankRepository(
        tmp_path / "metadata.sqlite3",
    )
    bank = build_bank()

    repository.save(bank)
    repository.save(bank)

    assert (
        repository.find(
            bank.document_id,
            bank.identity_fingerprint,
        )
        == bank
    )


def test_saving_conflicting_question_bank_is_rejected(
    tmp_path,
) -> None:
    repository = SqliteQuestionBankRepository(
        tmp_path / "metadata.sqlite3",
    )
    original = build_bank()
    conflicting = build_bank(
        questions=(
            build_question(
                text="A different question?",
            ),
        )
    )

    repository.save(original)

    with pytest.raises(
        ValueError,
        match="Conflicting question bank already exists",
    ):
        repository.save(conflicting)

    assert (
        repository.find(
            original.document_id,
            original.identity_fingerprint,
        )
        == original
    )


def test_replace_updates_existing_question_bank_explicitly(
    tmp_path,
) -> None:
    repository = SqliteQuestionBankRepository(
        tmp_path / "metadata.sqlite3",
    )
    original = build_bank()
    replacement = build_bank(
        questions=(
            build_question(
                text="Explicitly regenerated question?",
            ),
        )
    )

    repository.save(original)
    repository.replace(replacement)

    assert (
        repository.find(
            original.document_id,
            original.identity_fingerprint,
        )
        == replacement
    )


def test_list_document_returns_only_its_question_banks(
    tmp_path,
) -> None:
    repository = SqliteQuestionBankRepository(
        tmp_path / "metadata.sqlite3",
    )
    document_id = UUID("aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa")
    other_document_id = UUID(
        "bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb",
    )
    second_bank = build_bank(
        document_id=document_id,
        identity_fingerprint="2" * 64,
    )
    first_bank = build_bank(
        document_id=document_id,
        identity_fingerprint="1" * 64,
    )
    other_bank = build_bank(
        document_id=other_document_id,
        identity_fingerprint="3" * 64,
    )

    repository.save(second_bank)
    repository.save(first_bank)
    repository.save(other_bank)

    assert repository.list_document(document_id) == [
        first_bank,
        second_bank,
    ]


def test_delete_document_removes_all_its_question_banks(
    tmp_path,
) -> None:
    repository = SqliteQuestionBankRepository(
        tmp_path / "metadata.sqlite3",
    )
    document_id = UUID("aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa")
    other_document_id = UUID(
        "bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb",
    )
    first_bank = build_bank(
        document_id=document_id,
        identity_fingerprint="1" * 64,
    )
    second_bank = build_bank(
        document_id=document_id,
        identity_fingerprint="2" * 64,
    )
    other_bank = build_bank(
        document_id=other_document_id,
        identity_fingerprint="3" * 64,
    )
    repository.save(first_bank)
    repository.save(second_bank)
    repository.save(other_bank)

    deleted_count = repository.delete_document(document_id)

    assert deleted_count == 2
    assert repository.list_document(document_id) == []
    assert repository.list_document(other_document_id) == [
        other_bank,
    ]
