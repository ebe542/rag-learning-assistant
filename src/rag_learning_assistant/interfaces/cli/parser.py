"""Argument parsing and validation for the command-line interface."""

import argparse
from pathlib import Path
from uuid import UUID

DEFAULT_MAX_CHARS = 1000
DEFAULT_OVERLAP_CHARS = 150
DEFAULT_RESULT_LIMIT = 5
DEFAULT_SUMMARY_MAX_MAP_NEW_TOKENS = 192
DEFAULT_SUMMARY_MAX_REDUCE_NEW_TOKENS = 384
DEFAULT_SUMMARY_MAX_BATCH_CHARS = 12_000
DEFAULT_QUESTION_COUNT = 5
DEFAULT_QUESTION_MAX_NEW_TOKENS = 512


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

    allowed_entries = {
        "vectors.faiss",
        "metadata.sqlite3",
    }

    # A failed first import can leave valid metadata before vectors exist. The
    # directory must remain reusable, while a vector index without metadata is
    # incomplete and cannot be assigned to library documents safely.
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
    extract_parser.add_argument(
        "pdf",
        type=Path,
        help="Path to a text-based PDF",
    )
    add_chunking_options(extract_parser)

    index_parser = commands.add_parser(
        "index",
        help="Add one or more PDFs to a persistent library",
    )
    index_parser.add_argument(
        "pdfs",
        nargs="+",
        type=Path,
        help="Paths to text-based PDF documents",
    )
    add_chunking_options(index_parser)
    index_parser.add_argument(
        "--index-dir",
        type=Path,
        required=True,
        help="Directory for the FAISS index and SQLite metadata",
    )

    list_parser = commands.add_parser(
        "list",
        help="List documents in a persistent library",
    )
    list_parser.add_argument(
        "index_dir",
        type=Path,
        help="Directory containing the library metadata",
    )
    remove_parser = commands.add_parser(
        "remove",
        help="Remove a document from a persistent library",
    )
    remove_parser.add_argument(
        "document_id",
        type=UUID,
        help="UUID of the document to remove",
    )
    remove_parser.add_argument(
        "--index-dir",
        type=Path,
        required=True,
        help="Directory containing the library index",
    )
    replace_parser = commands.add_parser(
        "replace",
        help="Replace a library document while preserving its UUID",
    )
    replace_parser.add_argument(
        "document_id",
        type=UUID,
        help="UUID of the document to replace",
    )
    replace_parser.add_argument(
        "pdf",
        type=Path,
        help="Path to the replacement PDF",
    )
    replace_parser.add_argument(
        "--index-dir",
        type=Path,
        required=True,
        help="Directory containing the library index",
    )
    add_chunking_options(replace_parser)

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

    ask_parser = commands.add_parser(
        "ask",
        help="Answer a question using an existing persistent index",
    )
    ask_parser.add_argument(
        "index_dir",
        type=Path,
        help="Directory containing the FAISS index and SQLite metadata",
    )
    ask_parser.add_argument(
        "question",
        type=non_blank_text,
        help="Question to answer from the indexed documents",
    )
    ask_parser.add_argument(
        "--limit",
        type=positive_int,
        default=DEFAULT_RESULT_LIMIT,
        help=f"Maximum number of source contexts (default: {DEFAULT_RESULT_LIMIT})",
    )

    summarize_parser = commands.add_parser(
        "summarize",
        help="Summarize one document from an existing persistent index",
    )
    summarize_parser.add_argument(
        "index_dir",
        type=Path,
        help="Directory containing the library index",
    )
    summarize_parser.add_argument(
        "document_id",
        type=UUID,
        help="UUID of the document to summarize",
    )
    summarize_parser.add_argument(
        "--max-map-new-tokens",
        type=positive_int,
        default=DEFAULT_SUMMARY_MAX_MAP_NEW_TOKENS,
        help=(
            "Maximum tokens generated for each partial summary "
            f"(default: {DEFAULT_SUMMARY_MAX_MAP_NEW_TOKENS})"
        ),
    )
    summarize_parser.add_argument(
        "--max-reduce-new-tokens",
        type=positive_int,
        default=DEFAULT_SUMMARY_MAX_REDUCE_NEW_TOKENS,
        help=(
            "Maximum tokens generated for the final summary "
            f"(default: {DEFAULT_SUMMARY_MAX_REDUCE_NEW_TOKENS})"
        ),
    )
    summarize_parser.add_argument(
        "--max-batch-chars",
        type=positive_int,
        default=DEFAULT_SUMMARY_MAX_BATCH_CHARS,
        help=(
            "Maximum source characters per summary batch "
            f"(default: {DEFAULT_SUMMARY_MAX_BATCH_CHARS})"
        ),
    )
    summarize_parser.add_argument(
        "--force",
        action="store_true",
        help="Regenerate the summary even when a matching final result exists",
    )

    summary_list_parser = commands.add_parser(
        "summary-list",
        help="List persisted summaries for one library document",
    )
    summary_list_parser.add_argument(
        "index_dir",
        type=Path,
        help="Directory containing the library metadata",
    )
    summary_list_parser.add_argument(
        "document_id",
        type=UUID,
        help="UUID of the document whose summaries should be listed",
    )

    summary_show_parser = commands.add_parser(
        "summary-show",
        help="Show one persisted summary with citations",
    )
    summary_show_parser.add_argument(
        "index_dir",
        type=Path,
        help="Directory containing the library metadata",
    )
    summary_show_parser.add_argument(
        "document_id",
        type=UUID,
        help="UUID of the summarized document",
    )
    summary_show_parser.add_argument(
        "identity_fingerprint",
        type=sha256_fingerprint,
        help="SHA-256 generation identity of the stored summary",
    )

    question_generate_parser = commands.add_parser(
        "question-generate",
        help="Generate a grounded question bank from a persisted summary",
    )
    question_generate_parser.add_argument(
        "index_dir",
        type=Path,
        help="Directory containing the library metadata",
    )
    question_generate_parser.add_argument(
        "document_id",
        type=UUID,
        help="UUID of the source document",
    )
    question_generate_parser.add_argument(
        "summary_identity_fingerprint",
        type=sha256_fingerprint,
        help="SHA-256 identity of the persisted source summary",
    )
    question_generate_parser.add_argument(
        "--count",
        type=positive_int,
        default=DEFAULT_QUESTION_COUNT,
        help=(f"Number of study questions to generate (default: {DEFAULT_QUESTION_COUNT})"),
    )
    question_generate_parser.add_argument(
        "--max-new-tokens",
        type=positive_int,
        default=DEFAULT_QUESTION_MAX_NEW_TOKENS,
        help=(
            "Maximum tokens generated for the complete question bank "
            f"(default: {DEFAULT_QUESTION_MAX_NEW_TOKENS})"
        ),
    )
    question_generate_parser.add_argument(
        "--force",
        action="store_true",
        help="Regenerate and replace an existing matching question bank",
    )

    question_list_parser = commands.add_parser(
        "question-list",
        help="List persisted question banks for one document",
    )
    question_list_parser.add_argument(
        "index_dir",
        type=Path,
        help="Directory containing the library metadata",
    )
    question_list_parser.add_argument(
        "document_id",
        type=UUID,
        help="UUID of the document whose question banks should be listed",
    )

    question_show_parser = commands.add_parser(
        "question-show",
        help="Show one persisted grounded question bank",
    )
    question_show_parser.add_argument(
        "index_dir",
        type=Path,
        help="Directory containing the library metadata",
    )
    question_show_parser.add_argument(
        "document_id",
        type=UUID,
        help="UUID of the source document",
    )
    question_show_parser.add_argument(
        "identity_fingerprint",
        type=sha256_fingerprint,
        help="SHA-256 generation identity of the question bank",
    )
    return parser
