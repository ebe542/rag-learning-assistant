"""Compose the command-line parser while preserving its public API."""

import argparse

from rag_learning_assistant.interfaces.cli.parsers import (
    add_document_commands,
    add_package_commands,
    add_question_commands,
    add_retrieval_commands,
    add_review_commands,
    add_summary_commands,
)
from rag_learning_assistant.interfaces.cli.parsing import (
    DEFAULT_MAX_CHARS,
    DEFAULT_OVERLAP_CHARS,
    DEFAULT_QUESTION_COUNT,
    DEFAULT_QUESTION_MAX_NEW_TOKENS,
    DEFAULT_RESULT_LIMIT,
    DEFAULT_REVIEW_LIMIT,
    DEFAULT_SUMMARY_MAX_BATCH_CHARS,
    DEFAULT_SUMMARY_MAX_MAP_NEW_TOKENS,
    DEFAULT_SUMMARY_MAX_REDUCE_NEW_TOKENS,
    add_chunking_options,
    non_blank_text,
    positive_int,
    sha256_fingerprint,
    validate_existing_index_directory,
    validate_index_directory,
    validate_library_directory,
)

__all__ = [
    "DEFAULT_MAX_CHARS",
    "DEFAULT_OVERLAP_CHARS",
    "DEFAULT_QUESTION_COUNT",
    "DEFAULT_QUESTION_MAX_NEW_TOKENS",
    "DEFAULT_RESULT_LIMIT",
    "DEFAULT_REVIEW_LIMIT",
    "DEFAULT_SUMMARY_MAX_BATCH_CHARS",
    "DEFAULT_SUMMARY_MAX_MAP_NEW_TOKENS",
    "DEFAULT_SUMMARY_MAX_REDUCE_NEW_TOKENS",
    "add_chunking_options",
    "build_parser",
    "non_blank_text",
    "positive_int",
    "sha256_fingerprint",
    "validate_existing_index_directory",
    "validate_index_directory",
    "validate_library_directory",
]


def build_parser() -> argparse.ArgumentParser:
    """Build the top-level parser and delegate command registration."""

    parser = argparse.ArgumentParser(
        description="Build learning material from PDF documents",
    )
    commands = parser.add_subparsers(
        dest="command",
        required=True,
    )

    add_document_commands(commands)
    add_retrieval_commands(commands)
    add_summary_commands(commands)
    add_question_commands(commands)
    add_review_commands(commands)
    add_package_commands(commands)

    return parser
