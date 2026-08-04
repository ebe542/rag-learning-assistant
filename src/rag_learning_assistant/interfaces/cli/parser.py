"""Argument parsing and validation for the command-line interface."""

import argparse
from pathlib import Path

DEFAULT_MAX_CHARS = 1000
DEFAULT_OVERLAP_CHARS = 150
DEFAULT_RESULT_LIMIT = 5


def positive_int(value: str) -> int:
    """Parse a positive integer for argparse."""

    try:
        number = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("value must be an integer") from exc

    if number < 1:
        raise argparse.ArgumentTypeError("value must be positive")

    return number


def non_blank_text(value: str) -> str:
    """Reject empty command-line text values."""

    if not value.strip():
        raise argparse.ArgumentTypeError("value must not be blank")

    return value


def validate_empty_index_directory(index_directory: Path) -> None:
    """Require a new or empty directory for document indexing."""

    if not index_directory.exists():
        return

    if not index_directory.is_dir() or any(index_directory.iterdir()):
        raise ValueError("index directory must be empty")


def validate_existing_index_directory(index_directory: Path) -> None:
    """Require both persistent index files before starting retrieval."""

    required_files = (
        index_directory / "vectors.faiss",
        index_directory / "metadata.sqlite3",
    )

    if not index_directory.is_dir() or not all(path.is_file() for path in required_files):
        raise ValueError("index directory is incomplete")


def add_chunking_arguments(parser: argparse.ArgumentParser) -> None:
    """Add options shared by commands that process documents."""

    parser.add_argument("pdf", type=Path, help="Path to a text-based PDF")
    parser.add_argument(
        "--max-chars",
        type=int,
        default=DEFAULT_MAX_CHARS,
        help=f"Maximum characters per chunk (default: {DEFAULT_MAX_CHARS})",
    )
    parser.add_argument(
        "--overlap-chars",
        type=int,
        default=DEFAULT_OVERLAP_CHARS,
        help=f"Overlap within split paragraphs (default: {DEFAULT_OVERLAP_CHARS})",
    )


def build_parser() -> argparse.ArgumentParser:
    """Build the top-level parser and all supported subcommands."""

    parser = argparse.ArgumentParser(description="Build learning material from PDF documents")
    commands = parser.add_subparsers(
        dest="command",
        required=True,
    )

    extract_parser = commands.add_parser(
        "extract",
        help="Extract pages and chunks as JSON",
    )
    add_chunking_arguments(extract_parser)

    index_parser = commands.add_parser(
        "index",
        help="Create a persistent search index for a PDF",
    )
    add_chunking_arguments(index_parser)
    index_parser.add_argument(
        "--index-dir",
        type=Path,
        required=True,
        help="Directory for the FAISS index and SQLite metadata",
    )

    search_parser = commands.add_parser(
        "search",
        help="Search an existing persistent index",
    )
    search_parser.add_argument(
        "index_dir",
        type=Path,
        help="Directory containing the FAISS index and SQLite metadata",
    )
    search_parser.add_argument(
        "query",
        type=non_blank_text,
        help="Question or search text",
    )
    search_parser.add_argument(
        "--limit",
        type=positive_int,
        default=DEFAULT_RESULT_LIMIT,
        help=f"Maximum number of results (default: {DEFAULT_RESULT_LIMIT})",
    )

    return parser
