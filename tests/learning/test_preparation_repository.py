import sqlite3
from contextlib import closing
from datetime import UTC, datetime, timedelta
from pathlib import Path
from uuid import UUID

import pytest

from rag_learning_assistant.learning import (
    LearningLanguage,
    LearningPackage,
    LearningPackageStatus,
    PackagePreparation,
    PackagePreparationStatus,
    SqliteLearningPackageRepository,
    SqlitePackagePreparationRepository,
)

NOW = datetime(2026, 8, 29, 12, 0, tzinfo=UTC)


def pending(
    preparation_id: UUID,
    name: str,
    *,
    created_at: datetime = NOW,
    learning_language: LearningLanguage = LearningLanguage.SAME_AS_DOCUMENT,
) -> PackagePreparation:
    return PackagePreparation(
        id=preparation_id,
        name=name,
        source_filename=f"{name}.pdf",
        stored_filename=f"{preparation_id}.pdf",
        question_count=5,
        size_bytes=8,
        created_at=created_at,
        learning_language=learning_language,
    )


def test_learning_language_survives_preparation_repository_reopening(tmp_path: Path) -> None:
    database_path = tmp_path / "metadata.sqlite3"
    preparation = pending(
        UUID("11111111-1111-1111-1111-111111111111"),
        "German course",
        learning_language=LearningLanguage.GERMAN,
    )
    SqlitePackagePreparationRepository(database_path).save(preparation)

    reopened = SqlitePackagePreparationRepository(database_path)

    assert reopened.find_by_name(preparation.name) == preparation


def test_claim_is_atomic_and_uses_oldest_pending_request(tmp_path: Path) -> None:
    repository = SqlitePackagePreparationRepository(tmp_path / "metadata.sqlite3")
    older_id = UUID("11111111-1111-1111-1111-111111111111")
    newer_id = UUID("22222222-2222-2222-2222-222222222222")
    lease_token = UUID("aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa")
    repository.save(pending(newer_id, "Newer", created_at=NOW + timedelta(minutes=1)))
    repository.save(pending(older_id, "Older"))

    claimed = repository.claim_next(
        lease_token=lease_token,
        now=NOW + timedelta(minutes=2),
        lease_duration=timedelta(minutes=5),
    )

    assert claimed is not None
    assert claimed.id == older_id
    assert claimed.status is PackagePreparationStatus.INDEXING
    assert claimed.lease_token == lease_token
    second = repository.claim_next(
        lease_token=UUID("bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb"),
        now=NOW + timedelta(minutes=2),
        lease_duration=timedelta(minutes=5),
    )
    assert second is not None
    assert second.id == newer_id


def test_expired_lease_is_reclaimed_at_its_current_phase(tmp_path: Path) -> None:
    repository = SqlitePackagePreparationRepository(tmp_path / "metadata.sqlite3")
    preparation_id = UUID("11111111-1111-1111-1111-111111111111")
    first_token = UUID("aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa")
    second_token = UUID("bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb")
    repository.save(pending(preparation_id, "Course"))
    indexing = repository.claim_next(
        lease_token=first_token,
        now=NOW,
        lease_duration=timedelta(minutes=5),
    )
    assert indexing is not None
    summarizing = repository.advance(
        preparation_id,
        lease_token=first_token,
        current_status=PackagePreparationStatus.INDEXING,
        next_status=PackagePreparationStatus.SUMMARIZING,
        now=NOW + timedelta(minutes=1),
        lease_duration=timedelta(minutes=5),
    )

    assert summarizing.status is PackagePreparationStatus.SUMMARIZING
    assert (
        repository.claim_next(
            lease_token=second_token,
            now=NOW + timedelta(minutes=5),
            lease_duration=timedelta(minutes=5),
        )
        is None
    )
    reclaimed = repository.claim_next(
        lease_token=second_token,
        now=NOW + timedelta(minutes=7),
        lease_duration=timedelta(minutes=5),
    )
    assert reclaimed is not None
    assert reclaimed.status is PackagePreparationStatus.SUMMARIZING
    assert reclaimed.lease_token == second_token


def test_only_current_lease_can_fail_and_retry_a_request(tmp_path: Path) -> None:
    repository = SqlitePackagePreparationRepository(tmp_path / "metadata.sqlite3")
    preparation_id = UUID("11111111-1111-1111-1111-111111111111")
    lease_token = UUID("aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa")
    repository.save(pending(preparation_id, "Course"))
    repository.claim_next(
        lease_token=lease_token,
        now=NOW,
        lease_duration=timedelta(minutes=5),
    )

    with pytest.raises(ValueError, match="lease is no longer valid"):
        repository.mark_failed(
            preparation_id,
            lease_token=UUID("bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb"),
            now=NOW + timedelta(minutes=1),
            message="Index failed",
        )

    failed = repository.mark_failed(
        preparation_id,
        lease_token=lease_token,
        now=NOW + timedelta(minutes=1),
        message="Index failed",
    )
    assert failed.status is PackagePreparationStatus.FAILED
    assert failed.failure_message == "Index failed"
    retried = repository.retry_failed(preparation_id, now=NOW + timedelta(minutes=2))
    assert retried.status is PackagePreparationStatus.PENDING
    assert retried.failure_message is None


def test_renewed_lease_cannot_be_reclaimed_at_its_original_expiry(tmp_path: Path) -> None:
    repository = SqlitePackagePreparationRepository(tmp_path / "metadata.sqlite3")
    preparation_id = UUID("11111111-1111-1111-1111-111111111111")
    lease_token = UUID("aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa")
    repository.save(pending(preparation_id, "Course"))
    repository.claim_next(
        lease_token=lease_token,
        now=NOW,
        lease_duration=timedelta(minutes=5),
    )

    renewed = repository.renew_lease(
        preparation_id,
        lease_token=lease_token,
        now=NOW + timedelta(minutes=4),
        lease_duration=timedelta(minutes=5),
    )

    assert renewed.lease_expires_at == NOW + timedelta(minutes=9)
    assert (
        repository.claim_next(
            lease_token=UUID("bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb"),
            now=NOW + timedelta(minutes=6),
            lease_duration=timedelta(minutes=5),
        )
        is None
    )


def test_existing_pending_schema_is_migrated_in_place(tmp_path: Path) -> None:
    database_path = tmp_path / "metadata.sqlite3"
    preparation_id = UUID("11111111-1111-1111-1111-111111111111")
    with closing(sqlite3.connect(database_path)) as connection, connection:
        connection.execute(
            """
            CREATE TABLE package_preparations (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL UNIQUE COLLATE NOCASE,
                source_filename TEXT NOT NULL,
                stored_filename TEXT NOT NULL UNIQUE,
                question_count INTEGER NOT NULL,
                size_bytes INTEGER NOT NULL,
                status TEXT NOT NULL
            )
            """
        )
        connection.execute(
            "INSERT INTO package_preparations VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                str(preparation_id),
                "Existing",
                "existing.pdf",
                f"{preparation_id}.pdf",
                5,
                8,
                "pending",
            ),
        )

    repository = SqlitePackagePreparationRepository(database_path)
    migrated = repository.find_by_name("existing")

    assert migrated is not None
    assert migrated.status is PackagePreparationStatus.PENDING
    assert migrated.created_at.tzinfo is not None


def test_pending_name_reservation_blocks_materialized_package(tmp_path: Path) -> None:
    database_path = tmp_path / "metadata.sqlite3"
    preparation_repository = SqlitePackagePreparationRepository(database_path)
    preparation_repository.save(
        pending(UUID("11111111-1111-1111-1111-111111111111"), "Python Course")
    )
    package_repository = SqliteLearningPackageRepository(database_path)

    with pytest.raises(ValueError, match="already exists"):
        package_repository.save(
            LearningPackage(
                id=UUID("aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa"),
                name="python course",
                document_id=UUID("bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb"),
                status=LearningPackageStatus.INDEXED,
            )
        )


def test_indexed_checkpoint_can_reopen_with_its_preparation_request(tmp_path: Path) -> None:
    database_path = tmp_path / "metadata.sqlite3"
    preparation_id = UUID("11111111-1111-1111-1111-111111111111")
    preparation_repository = SqlitePackagePreparationRepository(database_path)
    preparation_repository.save(pending(preparation_id, "Python Course"))
    package_repository = SqliteLearningPackageRepository(database_path)
    package_repository.save_from_preparation(
        LearningPackage(
            id=UUID("aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa"),
            name="Python Course",
            document_id=UUID("bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb"),
            status=LearningPackageStatus.INDEXED,
        ),
        preparation_id,
    )

    reopened = SqlitePackagePreparationRepository(database_path)

    assert reopened.find_by_name("Python Course") is not None
