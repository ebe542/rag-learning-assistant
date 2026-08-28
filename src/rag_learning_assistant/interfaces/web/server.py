"""Run the local browser interface with a fixed security boundary."""

import threading
import webbrowser
from pathlib import Path

import uvicorn

from rag_learning_assistant.interfaces.web.application import create_app
from rag_learning_assistant.interfaces.web.libraries import LocalLibraryManager

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
    libraries = LocalLibraryManager(library_directory)
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
