"""Workspace loop for serial GUI package preparation."""

import logging
import threading
from collections.abc import Callable
from pathlib import Path

from rag_learning_assistant.application import (
    LearningPackageService,
    PackagePreparationService,
    PackagePreparationWorker,
)
from rag_learning_assistant.learning import SqlitePackagePreparationRepository

LOGGER = logging.getLogger(__name__)


class WorkspacePreparationWorker:
    """Process pending requests from every direct library in one workspace."""

    def __init__(
        self,
        workspace_directory: Path,
        package_service_factory: Callable[
            [Path, Callable[[str], None]],
            LearningPackageService,
        ],
        *,
        poll_interval: float = 1.0,
    ) -> None:
        self.workspace_directory = workspace_directory.resolve()
        self.package_service_factory = package_service_factory
        self.poll_interval = poll_interval

    def run(self, stop_event: threading.Event) -> None:
        """Poll until shutdown, processing only one package at a time."""

        while not stop_event.is_set():
            processed = False
            for library_directory in self._library_directories():
                if stop_event.is_set():
                    return
                try:
                    preparations = PackagePreparationService(
                        SqlitePackagePreparationRepository(library_directory / "metadata.sqlite3"),
                        library_directory / "uploads",
                    )
                    worker = PackagePreparationWorker(
                        preparations,
                        lambda progress, directory=library_directory: self.package_service_factory(
                            directory, progress
                        ),
                        library_directory / "uploads",
                    )
                    if worker.run_once():
                        processed = True
                        break
                except Exception:
                    LOGGER.exception(
                        "Package preparation worker failed for %s",
                        library_directory,
                    )
            if not processed:
                stop_event.wait(self.poll_interval)

    def _library_directories(self) -> tuple[Path, ...]:
        if not self.workspace_directory.is_dir():
            return ()
        return tuple(
            sorted(
                (
                    path
                    for path in self.workspace_directory.iterdir()
                    if path.is_dir() and (path / "metadata.sqlite3").is_file()
                ),
                key=lambda path: path.name.casefold(),
            )
        )
