"""Shared CLI parsing constants, validators, and options."""

import argparse
from pathlib import Path
from typing import Protocol

DEFAULT_MAX_CHARS = 1000
DEFAULT_OVERLAP_CHARS = 150
DEFAULT_RESULT_LIMIT = 5
DEFAULT_SUMMARY_MAX_MAP_NEW_TOKENS = 192
DEFAULT_SUMMARY_MAX_REDUCE_NEW_TOKENS = 384
DEFAULT_SUMMARY_MAX_BATCH_CHARS = 12_000
DEFAULT_QUESTION_COUNT = 5
DEFAULT_QUESTION_BATCH_SIZE = 3
DEFAULT_QUESTION_MAX_NEW_TOKENS = 512
DEFAULT_REVIEW_LIMIT = 10
DEFAULT_ANSWER_EVALUATION_MAX_NEW_TOKENS = 256


class SubcommandCollection(Protocol):
    """Expose the argparse operation needed by command registrars."""

    def add_parser(
        self,
        name: str,
        *,
        help: str | None = None,
    ) -> argparse.ArgumentParser: ...


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


def validate_index_directory(index_directory: Path) -> None:
    """Require a directory that can safely receive indexed documents."""

    if not index_directory.exists():
        return

    if not index_directory.is_dir():
        raise ValueError("index directory is incomplete")

    entries = {path.name for path in index_directory.iterdir()}
    if not entries:
        return

    allowed_entries = {"vectors.faiss", "metadata.sqlite3"}

    # A failed first import can leave valid metadata before vectors exist. The
    # directory must remain reusable, while vectors without metadata are unsafe.
    if not entries.issubset(allowed_entries) or entries == {"vectors.faiss"}:
        raise ValueError("index directory is incomplete")


def validate_existing_index_directory(index_directory: Path) -> None:
    """Require both persistent index files before starting retrieval."""

    required_files = (
        index_directory / "vectors.faiss",
        index_directory / "metadata.sqlite3",
    )

    if not index_directory.is_dir() or not all(path.is_file() for path in required_files):
        raise ValueError("index directory is incomplete")


def validate_library_directory(index_directory: Path) -> None:
    """Require the SQLite metadata of an existing library."""

    metadata_path = index_directory / "metadata.sqlite3"
    if not index_directory.is_dir() or not metadata_path.is_file():
        raise ValueError("library directory is incomplete")


def sha256_fingerprint(value: str) -> str:
    """Parse and normalize a SHA-256 hexadecimal fingerprint."""

    normalized = value.strip().lower()
    if len(normalized) != 64 or any(
        character not in "0123456789abcdef" for character in normalized
    ):
        raise argparse.ArgumentTypeError(
            "value must be a 64-character SHA-256 fingerprint",
        )

    return normalized


def add_chunking_options(parser: argparse.ArgumentParser) -> None:
    """Add options shared by commands that process documents."""

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
