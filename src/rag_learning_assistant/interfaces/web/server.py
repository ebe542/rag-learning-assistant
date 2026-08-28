"""Run the local browser interface with a fixed security boundary."""

import threading
import webbrowser
from pathlib import Path

import uvicorn

from rag_learning_assistant.application import LearningPackageCatalog
from rag_learning_assistant.interfaces.web.application import create_app
from rag_learning_assistant.learning import SqliteLearningPackageRepository

LOOPBACK_HOST = "127.0.0.1"


def run_server(
    library_directory: Path,
    port: int,
    *,
    open_browser: bool,
) -> None:
    """Serve the GUI on loopback and optionally open its start page."""

    url = f"http://{LOOPBACK_HOST}:{port}"
    if open_browser:
        browser_timer = threading.Timer(0.75, webbrowser.open, args=(url,))
        browser_timer.daemon = True
        browser_timer.start()

    print(f"RAG Learning Assistant GUI: {url}", flush=True)
    packages = LearningPackageCatalog(
        SqliteLearningPackageRepository(library_directory / "metadata.sqlite3")
    )
    uvicorn.run(
        create_app(packages, library_directory=library_directory),
        host=LOOPBACK_HOST,
        port=port,
        access_log=False,
    )
