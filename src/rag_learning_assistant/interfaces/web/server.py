"""Run the local browser interface with a fixed security boundary."""

import threading
import webbrowser
from collections.abc import Callable
from pathlib import Path

import uvicorn

from rag_learning_assistant.application import LearningPackageService
from rag_learning_assistant.interfaces.web.application import create_app
from rag_learning_assistant.interfaces.web.libraries import LocalLibraryManager
from rag_learning_assistant.interfaces.web.preparation_worker import WorkspacePreparationWorker

LOOPBACK_HOST = "127.0.0.1"


def run_server(
    library_directory: Path,
    port: int,
    *,
    open_browser: bool,
    package_service_factory: Callable[
        [Path, Callable[[str], None]],
        LearningPackageService,
    ],
    package_remover: Callable[[Path, str], None],
) -> None:
    """Serve the GUI on loopback and optionally open its start page."""

    url = f"http://{LOOPBACK_HOST}:{port}"
    if open_browser:
        browser_timer = threading.Timer(0.75, webbrowser.open, args=(url,))
        browser_timer.daemon = True
        browser_timer.start()

    print(f"RAG Learning Assistant GUI: {url}", flush=True)
    libraries = LocalLibraryManager(
        library_directory,
        package_remover=package_remover,
    )
    stop_worker = threading.Event()
    worker = WorkspacePreparationWorker(
        libraries.root_directory,
        package_service_factory,
    )
    worker_thread = threading.Thread(
        target=worker.run,
        args=(stop_worker,),
        name="package-preparation-worker",
        daemon=True,
    )
    worker_thread.start()
    try:
        uvicorn.run(
            create_app(
                libraries,
                libraries,
                libraries,
                libraries,
                libraries,
                libraries=libraries,
            ),
            host=LOOPBACK_HOST,
            port=port,
            access_log=False,
        )
    finally:
        stop_worker.set()
        worker_thread.join(timeout=5)
