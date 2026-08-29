from io import BytesIO
from pathlib import Path
from uuid import UUID

from rag_learning_assistant.application.package_preparation import PackagePreparationService
from rag_learning_assistant.learning import SqlitePackagePreparationRepository


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
        source=BytesIO(b"%PDF-1.7"),
    )

    assert repository.find_by_name("python course") == stored
