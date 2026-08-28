"""Register the local graphical-interface command."""

import argparse
from pathlib import Path

from rag_learning_assistant.interfaces.cli.parsing import (
    SubcommandCollection,
    default_library_directory,
)

DEFAULT_GUI_PORT = 8765


def tcp_port(value: str) -> int:
    """Parse a valid user-selectable TCP port."""

    try:
        port = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("Port must be an integer") from error
    if not 1 <= port <= 65535:
        raise argparse.ArgumentTypeError("Port must be between 1 and 65535")
    return port


def add_gui_command(commands: SubcommandCollection) -> None:
    """Register the local browser interface."""

    gui_parser = commands.add_parser(
        "gui",
        help="Open the local graphical interface",
    )
    gui_parser.add_argument(
        "--library",
        type=Path,
        default=default_library_directory(),
        help="Personal learning library (default: platform user-data directory)",
    )
    gui_parser.add_argument(
        "--port",
        type=tcp_port,
        default=DEFAULT_GUI_PORT,
        help=f"Local HTTP port (default: {DEFAULT_GUI_PORT})",
    )
    gui_parser.add_argument(
        "--no-browser",
        action="store_true",
        help="Start the interface without opening a browser",
    )
