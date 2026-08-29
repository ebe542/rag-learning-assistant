from pathlib import Path
from uuid import UUID

import pytest

from rag_learning_assistant.interfaces.web.libraries import LocalLibraryManager


def test_manager_creates_and_switches_between_isolated_libraries(tmp_path: Path) -> None:
    initial_directory = tmp_path / "library"
    legacy_id = UUID("11111111-1111-1111-1111-111111111111")
    created_id = UUID("22222222-2222-2222-2222-222222222222")
    ids = iter((legacy_id, created_id))
    manager = LocalLibraryManager(initial_directory, id_factory=ids.__next__)

    created = manager.create_library("German History")

    assert created.id == created_id
    assert created.name == "German History"
    assert created.directory == (tmp_path / str(created_id)).resolve()
    assert (created.directory / "metadata.sqlite3").is_file()
    assert manager.current_directory == initial_directory.resolve()
    assert manager.list_packages() == []

    selected = manager.select_library(legacy_id)

    assert selected.directory == initial_directory.resolve()
    assert manager.current_directory == initial_directory.resolve()
    assert [(item.id, item.name) for item in manager.list_libraries()] == [
        (created_id, "German History"),
        (legacy_id, "library"),
    ]
    assert (initial_directory / "library.json").is_file()


def test_display_name_does_not_determine_the_directory(tmp_path: Path) -> None:
    ids = iter(
        (
            UUID("11111111-1111-1111-1111-111111111111"),
            UUID("22222222-2222-2222-2222-222222222222"),
        )
    )
    manager = LocalLibraryManager(tmp_path / "library", id_factory=ids.__next__)

    created = manager.create_library("C++ / Python: Basics?")

    assert created.name == "C++ / Python: Basics?"
    assert created.directory.name == "22222222-2222-2222-2222-222222222222"
    assert not (tmp_path / "C++ / Python: Basics?").exists()


@pytest.mark.parametrize("name", ["", "   ", "bad\nname", "x" * 101])
def test_manager_rejects_invalid_display_names(tmp_path: Path, name: str) -> None:
    manager = LocalLibraryManager(tmp_path / "library")

    with pytest.raises(ValueError):
        manager.create_library(name)


def test_manager_migrates_an_existing_library_without_moving_it(tmp_path: Path) -> None:
    initial_directory = tmp_path / "existing-folder"
    library_id = UUID("11111111-1111-1111-1111-111111111111")

    manager = LocalLibraryManager(initial_directory, id_factory=lambda: library_id)

    assert manager.current_library.id == library_id
    assert manager.current_library.name == "existing-folder"
    assert manager.current_directory == initial_directory.resolve()
    assert (initial_directory / "library.json").is_file()


def test_manager_rejects_a_duplicate_name_case_insensitively(tmp_path: Path) -> None:
    manager = LocalLibraryManager(tmp_path / "library")
    manager.create_library("German History")

    with pytest.raises(ValueError, match="already exists"):
        manager.create_library("german history")


def test_manager_renames_only_display_metadata(tmp_path: Path) -> None:
    manager = LocalLibraryManager(tmp_path / "library")
    created = manager.create_library("Original name")

    renamed = manager.rename_library(created.id, "Renamed / display name")

    assert renamed.name == "Renamed / display name"
    assert renamed.directory == created.directory
    assert created.directory.name == str(created.id)


def test_manager_deletes_an_empty_library_after_name_confirmation(tmp_path: Path) -> None:
    manager = LocalLibraryManager(tmp_path / "library")
    created = manager.create_library("Temporary")

    manager.delete_library(
        created.id,
        confirmation="Temporary",
        delete_contents=False,
    )

    assert not created.directory.exists()


def test_manager_requires_extra_confirmation_for_non_empty_library(tmp_path: Path) -> None:
    manager = LocalLibraryManager(tmp_path / "library")
    created = manager.create_library("With content")
    (created.directory / "vectors.faiss").write_bytes(b"data")

    with pytest.raises(ValueError, match="all library contents"):
        manager.delete_library(
            created.id,
            confirmation="With content",
            delete_contents=False,
        )

    assert created.directory.exists()

    manager.delete_library(
        created.id,
        confirmation="With content",
        delete_contents=True,
    )

    assert not created.directory.exists()


def test_manager_keeps_an_empty_workspace_after_deleting_last_library(tmp_path: Path) -> None:
    initial_directory = tmp_path / "library"
    manager = LocalLibraryManager(initial_directory)
    initial_library = manager.current_library
    assert initial_library is not None

    manager.delete_library(
        initial_library.id,
        confirmation=initial_library.name,
        delete_contents=False,
    )

    assert manager.current_library is None
    assert manager.current_directory is None
    assert manager.list_libraries() == ()

    restarted_manager = LocalLibraryManager(initial_directory)

    assert restarted_manager.current_library is None
    assert restarted_manager.list_libraries() == ()


def test_first_library_created_in_empty_workspace_is_opened_explicitly(tmp_path: Path) -> None:
    initial_directory = tmp_path / "library"
    manager = LocalLibraryManager(initial_directory)
    initial_library = manager.current_library
    assert initial_library is not None
    manager.delete_library(
        initial_library.id,
        confirmation=initial_library.name,
        delete_contents=False,
    )

    created = manager.create_library("New start")

    assert manager.current_library is None
    restarted_manager = LocalLibraryManager(initial_directory)
    assert restarted_manager.current_library is None
    assert restarted_manager.list_libraries() == (created,)
    manager.select_library(created.id)
    assert manager.current_library == created
