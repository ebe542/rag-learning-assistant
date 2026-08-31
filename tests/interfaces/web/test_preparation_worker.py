from collections.abc import Callable
from pathlib import Path
from threading import Event

from rag_learning_assistant.application import LearningPackageService
from rag_learning_assistant.interfaces.web import preparation_worker
from rag_learning_assistant.interfaces.web.preparation_worker import WorkspacePreparationWorker


class StopAfterWait(Event):
    def wait(self, timeout: float | None = None) -> bool:
        self.set()
        return True


def test_worker_writes_unexpected_failures_to_the_application_log(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    library_directory = tmp_path / "library"
    library_directory.mkdir()
    (library_directory / "metadata.sqlite3").touch()
    recorded: list[tuple[Exception, str, dict[str, object]]] = []

    def fail_repository(database_path: Path) -> None:
        raise ValueError(f"Broken database: {database_path}")

    def record_error(
        error: Exception,
        *,
        command: str,
        context: dict[str, object],
    ) -> Path:
        recorded.append((error, command, context))
        return tmp_path / "application.log"

    monkeypatch.setattr(
        preparation_worker,
        "SqlitePackagePreparationRepository",
        fail_repository,
    )
    monkeypatch.setattr(preparation_worker, "write_exception_log", record_error)

    def unused_factory(
        directory: Path,
        progress: Callable[[str], None],
    ) -> LearningPackageService:
        raise AssertionError("A broken repository must fail before service construction")

    worker = WorkspacePreparationWorker(
        tmp_path,
        unused_factory,
    )

    worker.run(StopAfterWait())

    assert len(recorded) == 1
    error, command, context = recorded[0]
    assert isinstance(error, ValueError)
    assert command == "gui-package-worker"
    assert context == {"library_directory": library_directory}
    assert "Package preparation worker failed." in capsys.readouterr().err


def test_package_failure_reports_log_path_without_a_traceback(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    log_path = tmp_path / "application.log"
    monkeypatch.setattr(
        preparation_worker,
        "write_exception_log",
        lambda error, **kwargs: log_path,
    )

    preparation_worker._report_failure(
        RuntimeError("Model failed"),
        library_directory=tmp_path / "library",
        package_name="Python Course",
    )

    console = capsys.readouterr().err
    assert "Package preparation failed for Python Course." in console
    assert str(log_path) in console
    assert "Traceback" not in console
