from pathlib import Path
from uuid import UUID

from rag_learning_assistant.learning import (
    LearningLanguage,
    LearningPackage,
    LearningPackageStatus,
    SqliteLearningPackageRepository,
)


def test_learning_package_survives_repository_reopening(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "metadata.sqlite3"
    package = LearningPackage(
        id=UUID("aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa"),
        name="python-basics",
        document_id=UUID("bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb"),
        status=LearningPackageStatus.READY,
        summary_identity_fingerprint="c" * 64,
        question_bank_identity_fingerprint="d" * 64,
        learning_language=LearningLanguage.GERMAN,
    )

    SqliteLearningPackageRepository(database_path).save(package)

    reopened = SqliteLearningPackageRepository(database_path)

    assert reopened.find_by_name("python-basics") == package


def test_replace_advances_learning_package_status(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "metadata.sqlite3"
    repository = SqliteLearningPackageRepository(database_path)
    package_id = UUID("aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa")
    document_id = UUID("bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb")

    repository.save(
        LearningPackage(
            id=package_id,
            name="python-basics",
            document_id=document_id,
            status=LearningPackageStatus.INDEXED,
        )
    )

    summarized = LearningPackage(
        id=package_id,
        name="python-basics",
        document_id=document_id,
        status=LearningPackageStatus.SUMMARIZED,
        summary_identity_fingerprint="c" * 64,
    )
    repository.replace(summarized)

    assert repository.find_by_name("python-basics") == summarized


def test_saving_identical_learning_package_is_idempotent(
    tmp_path: Path,
) -> None:
    repository = SqliteLearningPackageRepository(tmp_path / "metadata.sqlite3")
    package = LearningPackage(
        id=UUID("aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa"),
        name="python-basics",
        document_id=UUID("bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb"),
        status=LearningPackageStatus.INDEXED,
    )

    repository.save(package)
    repository.save(package)

    assert repository.find_by_name("python-basics") == package


def test_list_all_returns_packages_by_name(
    tmp_path: Path,
) -> None:
    repository = SqliteLearningPackageRepository(tmp_path / "metadata.sqlite3")
    repository.save(
        LearningPackage(
            id=UUID("aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa"),
            name="Python Advanced",
            document_id=UUID("bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb"),
            status=LearningPackageStatus.INDEXED,
        )
    )
    repository.save(
        LearningPackage(
            id=UUID("cccccccc-cccc-cccc-cccc-cccccccccccc"),
            name="Algorithms",
            document_id=UUID("dddddddd-dddd-dddd-dddd-dddddddddddd"),
            status=LearningPackageStatus.INDEXED,
        )
    )

    packages = repository.list_all()

    assert [package.name for package in packages] == [
        "Algorithms",
        "Python Advanced",
    ]


def test_delete_document_removes_its_learning_package(
    tmp_path: Path,
) -> None:
    repository = SqliteLearningPackageRepository(tmp_path / "metadata.sqlite3")
    removed = LearningPackage(
        id=UUID("aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa"),
        name="Python Basics",
        document_id=UUID("bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb"),
        status=LearningPackageStatus.INDEXED,
    )
    retained = LearningPackage(
        id=UUID("cccccccc-cccc-cccc-cccc-cccccccccccc"),
        name="Algorithms",
        document_id=UUID("dddddddd-dddd-dddd-dddd-dddddddddddd"),
        status=LearningPackageStatus.INDEXED,
    )
    repository.save(removed)
    repository.save(retained)

    deleted_count = repository.delete_document(removed.document_id)

    assert deleted_count == 1
    assert repository.find_by_name("Python Basics") is None
    assert repository.find_by_name("Algorithms") == retained
