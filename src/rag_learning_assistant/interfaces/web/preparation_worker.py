"""Workspace loop for serial GUI package preparation."""

import sys
import threading
from collections.abc import Callable
from pathlib import Path

from rag_learning_assistant.application import (
    LearningPackageService,
    PackagePreparationService,
    PackagePreparationWorker,
)
from rag_learning_assistant.interfaces.cli.error_reporting import write_exception_log
from rag_learning_assistant.learning import SqlitePackagePreparationRepository


def _report_failure(
    error: Exception,
    *,
    library_directory: Path,
    package_name: str | None = None,
) -> None:
    """Log a technical worker failure and keep console output user-facing."""

    try:
        log_path = write_exception_log(
            error,
            command="gui-package-worker",
            context={
                "library_directory": library_directory,
                **({"package_name": package_name} if package_name is not None else {}),
            },
        )
    except Exception:
        print(
            "Package preparation failed. The diagnostic log could not be written.",
            file=sys.stderr,
            flush=True,
        )
        return

    subject = (
        f"Package preparation failed for {package_name}."
        if package_name
        else ("Package preparation worker failed.")
    )
    print(f"{subject} Details were written to: {log_path}", file=sys.stderr, flush=True)


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
                        failure_reporter=(
                            lambda error, preparation, directory=library_directory: _report_failure(
                                error,
                                library_directory=directory,
                                package_name=preparation.name,
                            )
                        ),
                    )
                    if worker.run_once():
                        processed = True
                        break
                except Exception as error:
                    _report_failure(
                        error,
                        library_directory=library_directory,
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
