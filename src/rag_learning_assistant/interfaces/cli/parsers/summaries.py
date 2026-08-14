"""Register persistent document-summarization commands."""

from pathlib import Path
from uuid import UUID

from rag_learning_assistant.interfaces.cli.parsing import (
    DEFAULT_SUMMARY_MAX_BATCH_CHARS,
    DEFAULT_SUMMARY_MAX_MAP_NEW_TOKENS,
    DEFAULT_SUMMARY_MAX_REDUCE_NEW_TOKENS,
    SubcommandCollection,
    positive_int,
    sha256_fingerprint,
)


def add_summary_commands(commands: SubcommandCollection) -> None:
    """Register generation and read-only summary commands."""

    summarize_parser = commands.add_parser(
        "summarize", help="Summarize one document from an existing persistent index"
    )
    summarize_parser.add_argument(
        "index_dir", type=Path, help="Directory containing the library index"
    )
    summarize_parser.add_argument(
        "document_id", type=UUID, help="UUID of the document to summarize"
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
        "summary-list", help="List persisted summaries for one library document"
    )
    summary_list_parser.add_argument(
        "index_dir", type=Path, help="Directory containing the library metadata"
    )
    summary_list_parser.add_argument(
        "document_id",
        type=UUID,
        help="UUID of the document whose summaries should be listed",
    )

    summary_show_parser = commands.add_parser(
        "summary-show", help="Show one persisted summary with citations"
    )
    summary_show_parser.add_argument(
        "index_dir", type=Path, help="Directory containing the library metadata"
    )
    summary_show_parser.add_argument(
        "document_id", type=UUID, help="UUID of the summarized document"
    )
    summary_show_parser.add_argument(
        "identity_fingerprint",
        type=sha256_fingerprint,
        help="SHA-256 generation identity of the stored summary",
    )
