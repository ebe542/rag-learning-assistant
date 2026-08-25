"""Register persistent grounded question-bank commands."""

from pathlib import Path
from uuid import UUID

from rag_learning_assistant.interfaces.cli.parsing import (
    DEFAULT_QUESTION_COUNT,
    DEFAULT_QUESTION_MAX_NEW_TOKENS,
    SubcommandCollection,
    positive_int,
    sha256_fingerprint,
)


def add_question_commands(commands: SubcommandCollection) -> None:
    """Register generation and read-only question-bank commands."""

    generate_parser = commands.add_parser(
        "question-generate",
        help="Generate a grounded question bank from a persisted summary",
    )
    generate_parser.add_argument(
        "index_dir", type=Path, help="Directory containing the library metadata"
    )
    generate_parser.add_argument("document_id", type=UUID, help="UUID of the source document")
    generate_parser.add_argument(
        "summary_identity_fingerprint",
        type=sha256_fingerprint,
        help="SHA-256 identity of the persisted source summary",
    )
    generate_parser.add_argument(
        "--count",
        type=positive_int,
        default=DEFAULT_QUESTION_COUNT,
        help=f"Number of study questions to generate (default: {DEFAULT_QUESTION_COUNT})",
    )
    generate_parser.add_argument(
        "--max-new-tokens",
        type=positive_int,
        default=DEFAULT_QUESTION_MAX_NEW_TOKENS,
        help=(
            "Maximum tokens generated for each question batch "
            f"(default: {DEFAULT_QUESTION_MAX_NEW_TOKENS})"
        ),
    )
    generate_parser.add_argument(
        "--force",
        action="store_true",
        help="Regenerate and replace an existing matching question bank",
    )

    list_parser = commands.add_parser(
        "question-list", help="List persisted question banks for one document"
    )
    list_parser.add_argument(
        "index_dir", type=Path, help="Directory containing the library metadata"
    )
    list_parser.add_argument(
        "document_id",
        type=UUID,
        help="UUID of the document whose question banks should be listed",
    )

    show_parser = commands.add_parser(
        "question-show", help="Show one persisted grounded question bank"
    )
    show_parser.add_argument(
        "index_dir", type=Path, help="Directory containing the library metadata"
    )
    show_parser.add_argument("document_id", type=UUID, help="UUID of the source document")
    show_parser.add_argument(
        "identity_fingerprint",
        type=sha256_fingerprint,
        help="SHA-256 generation identity of the question bank",
    )
