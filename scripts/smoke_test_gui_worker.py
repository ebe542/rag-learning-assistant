"""Create a disposable GUI smoke workspace and start the application."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from rag_learning_assistant.interfaces.cli.commands import run_gui
from rag_learning_assistant.interfaces.web.libraries import LocalLibraryManager
from rag_learning_assistant.learning import LearningLanguage

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Search these directories in order when resolving package fixture filenames.
TEST_DIRECTORIES = [
    PROJECT_ROOT / "local-data" / "books",
]

LIBRARY_NAME = "My Lib"
WORKSPACE_PREFIX = "smoke-gui-worker"
QUESTION_COUNT = 5
GUI_PORT = 8765
OPEN_BROWSER = True


@dataclass(frozen=True, slots=True)
class PackageFixture:
    """Describe one package queued before the GUI starts."""

    name: str
    filename: str
    learning_language: LearningLanguage


PACKAGE_FIXTURES: list[PackageFixture] = [
    # Add fixtures here.
]


def resolve_fixture(filename: str) -> Path:
    """Find one configured fixture without silently selecting a missing file."""

    matches = [
        directory / filename for directory in TEST_DIRECTORIES if (directory / filename).is_file()
    ]
    if not matches:
        searched = ", ".join(str(directory) for directory in TEST_DIRECTORIES)
        raise FileNotFoundError(f"Fixture not found: {filename}; searched: {searched}")
    if len(matches) > 1:
        locations = ", ".join(str(path) for path in matches)
        raise ValueError(f"Fixture is ambiguous: {filename}; matches: {locations}")
    return matches[0]


def calculate_sha256(path: Path) -> str:
    """Hash a fixture without loading the complete PDF into memory."""

    digest = hashlib.sha256()
    with path.open("rb") as source:
        while block := source.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    """Create the configured smoke data and run the normal loopback GUI."""

    resolved_fixtures = [
        (fixture, resolve_fixture(fixture.filename)) for fixture in PACKAGE_FIXTURES
    ]
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    workspace = PROJECT_ROOT / "local-data" / f"{WORKSPACE_PREFIX}-{timestamp}"
    workspace.mkdir(parents=False, exist_ok=False)

    # LocalLibraryManager treats the parent of its initial directory as the
    # workspace. The placeholder itself is intentionally never created.
    manager = LocalLibraryManager(workspace / "initial-library-placeholder")
    library = manager.create_library(LIBRARY_NAME)
    manager.select_library(library.id)

    print(f"Smoke workspace: {workspace}")
    print(f"Library: {library.name} ({library.directory})")

    for fixture, path in resolved_fixtures:
        with path.open("rb") as source:
            manager.store_package_upload(
                name=fixture.name,
                source_filename=path.name,
                question_count=QUESTION_COUNT,
                size_bytes=path.stat().st_size,
                content_sha256=calculate_sha256(path),
                source=source,
                learning_language=fixture.learning_language,
            )
        print(
            f"Queued: {fixture.name} <- {path.name} "
            f"(learning_language={fixture.learning_language.value})"
        )

    return run_gui(
        library.directory,
        GUI_PORT,
        open_browser=OPEN_BROWSER,
    )


if __name__ == "__main__":
    raise SystemExit(main())
