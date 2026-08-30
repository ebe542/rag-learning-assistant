from io import BytesIO
from pathlib import Path
from uuid import UUID

import pytest

from rag_learning_assistant.application.package_preparation import PackagePreparationService
from rag_learning_assistant.learning import (
    LearningPackage,
    LearningPackageStatus,
    SqliteLearningPackageRepository,
    SqlitePackagePreparationRepository,
)


def test_service_stores_upload_under_internal_id_and_persists_request(
    tmp_path: Path,
) -> None:
    preparation_id = UUID("11111111-1111-1111-1111-111111111111")
    repository = SqlitePackagePreparationRepository(tmp_path / "metadata.sqlite3")
    service = PackagePreparationService(
        repository,
        tmp_path / "uploads",
        id_factory=lambda: preparation_id,
    )

    stored = service.store(
        name="Python Course",
        source_filename="C:\\private\\course.pdf",
        question_count=7,
        size_bytes=12,
        content_sha256="a" * 64,
        source=BytesIO(b"%PDF-1.7\nxx"),
    )

    assert stored.source_filename == "course.pdf"
    assert stored.stored_filename == f"{preparation_id}.pdf"
    assert (tmp_path / "uploads" / stored.stored_filename).read_bytes() == b"%PDF-1.7\nxx"
    assert service.list_all() == [stored]


def test_repository_matches_pending_names_case_insensitively(tmp_path: Path) -> None:
    preparation_id = UUID("11111111-1111-1111-1111-111111111111")
    repository = SqlitePackagePreparationRepository(tmp_path / "metadata.sqlite3")
    service = PackagePreparationService(
        repository,
        tmp_path / "uploads",
        id_factory=lambda: preparation_id,
    )
    stored = service.store(
        name="Python Course",
        source_filename="course.pdf",
        question_count=5,
        size_bytes=8,
        content_sha256="a" * 64,
        source=BytesIO(b"%PDF-1.7"),
    )

    assert repository.find_by_name("python course") == stored


def test_service_rejects_duplicate_content_before_writing_another_upload(
    tmp_path: Path,
) -> None:
    ids = iter(
        (
            UUID("11111111-1111-1111-1111-111111111111"),
            UUID("22222222-2222-2222-2222-222222222222"),
        )
    )
    service = PackagePreparationService(
        SqlitePackagePreparationRepository(tmp_path / "metadata.sqlite3"),
        tmp_path / "uploads",
        id_factory=ids.__next__,
    )
    service.store(
        name="First name",
        source_filename="course.pdf",
        question_count=5,
        size_bytes=8,
        content_sha256="a" * 64,
        source=BytesIO(b"%PDF-1.7"),
    )

    with pytest.raises(ValueError, match="already queued as First name"):
        service.store(
            name="Different name",
            source_filename="renamed.pdf",
            question_count=5,
            size_bytes=8,
            content_sha256="a" * 64,
            source=BytesIO(b"%PDF-1.7"),
        )

    assert len(list((tmp_path / "uploads").iterdir())) == 1


def test_shared_name_reservation_removes_upload_when_package_exists(tmp_path: Path) -> None:
    database_path = tmp_path / "metadata.sqlite3"
    package_repository = SqliteLearningPackageRepository(database_path)
    package_repository.save(
        LearningPackage(
            id=UUID("aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa"),
            name="Python Course",
            document_id=UUID("bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb"),
            status=LearningPackageStatus.INDEXED,
        )
    )
    service = PackagePreparationService(
        SqlitePackagePreparationRepository(database_path),
        tmp_path / "uploads",
        id_factory=lambda: UUID("11111111-1111-1111-1111-111111111111"),
    )

    with pytest.raises(ValueError, match="already exists"):
        service.store(
            name="python course",
            source_filename="course.pdf",
            question_count=5,
            size_bytes=8,
            content_sha256="a" * 64,
            source=BytesIO(b"%PDF-1.7"),
        )

    assert list((tmp_path / "uploads").iterdir()) == []
