import sqlite3
from contextlib import closing
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


def test_list_all_returns_packages_by_arrival_time(
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

    with closing(sqlite3.connect(tmp_path / "metadata.sqlite3")) as connection, connection:
        connection.execute(
            "UPDATE learning_packages SET created_at = ? WHERE name = ?",
            ("2026-09-03T10:00:00+00:00", "Python Advanced"),
        )
        connection.execute(
            "UPDATE learning_packages SET created_at = ? WHERE name = ?",
            ("2026-09-03T11:00:00+00:00", "Algorithms"),
        )

    packages = repository.list_all()

    assert [package.name for package in packages] == [
        "Python Advanced",
        "Algorithms",
    ]


def test_existing_package_schema_receives_arrival_timestamps(tmp_path: Path) -> None:
    database_path = tmp_path / "metadata.sqlite3"
    with closing(sqlite3.connect(database_path)) as connection, connection:
        connection.execute(
            """
            CREATE TABLE learning_packages (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL UNIQUE COLLATE NOCASE,
                document_id TEXT NOT NULL UNIQUE,
                status TEXT NOT NULL,
                summary_identity_fingerprint TEXT,
                question_bank_identity_fingerprint TEXT,
                learning_language TEXT NOT NULL DEFAULT 'same'
            )
            """
        )
        connection.execute(
            """
            INSERT INTO learning_packages (
                id, name, document_id, status, learning_language
            ) VALUES (?, ?, ?, ?, ?)
            """,
            (
                "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa",
                "Existing",
                "bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb",
                "indexed",
                "same",
            ),
        )

    repository = SqliteLearningPackageRepository(database_path)

    assert [package.name for package in repository.list_all()] == ["Existing"]
    with closing(sqlite3.connect(database_path)) as connection:
        row = connection.execute(
            "SELECT created_at FROM learning_packages WHERE name = 'Existing'"
        ).fetchone()
    assert row is not None
    assert row[0] is not None


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
