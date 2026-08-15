"""Register spaced-review commands."""

from pathlib import Path
from uuid import UUID

from rag_learning_assistant.interfaces.cli.parsing import (
    DEFAULT_REVIEW_LIMIT,
    SubcommandCollection,
    positive_int,
    sha256_fingerprint,
)
from rag_learning_assistant.learning import ReviewRating


def add_review_commands(commands: SubcommandCollection) -> None:
    """Register due-question lookup and review recording commands."""

    due_parser = commands.add_parser(
        "review-due",
        help="List due study questions from one persisted question bank",
    )
    due_parser.add_argument(
        "index_dir", type=Path, help="Directory containing the library metadata"
    )
    due_parser.add_argument("document_id", type=UUID, help="UUID of the source document")
    due_parser.add_argument(
        "question_bank_identity_fingerprint",
        type=sha256_fingerprint,
        help="SHA-256 identity of the persisted question bank",
    )
    due_parser.add_argument(
        "--limit",
        type=positive_int,
        default=DEFAULT_REVIEW_LIMIT,
        help=f"Maximum number of due questions (default: {DEFAULT_REVIEW_LIMIT})",
    )

    record_parser = commands.add_parser(
        "review-record",
        help="Record a learner rating for one study question",
    )
    record_parser.add_argument(
        "index_dir", type=Path, help="Directory containing the library metadata"
    )
    record_parser.add_argument("document_id", type=UUID, help="UUID of the source document")
    record_parser.add_argument(
        "question_bank_identity_fingerprint",
        type=sha256_fingerprint,
        help="SHA-256 identity of the persisted question bank",
    )
    record_parser.add_argument(
        "question_number", type=positive_int, help="Number of the reviewed question"
    )
    record_parser.add_argument(
        "rating",
        type=ReviewRating,
        choices=tuple(ReviewRating),
        help="Learner rating: again, hard, good, or easy",
    )

    study_parser = commands.add_parser(
        "study",
        help="Interactively answer the next due study question",
    )
    study_parser.add_argument(
        "index_dir",
        type=Path,
        help="Directory containing the library metadata",
    )
    study_parser.add_argument(
        "document_id",
        type=UUID,
        help="UUID of the source document",
    )
    study_parser.add_argument(
        "question_bank_identity_fingerprint",
        type=sha256_fingerprint,
        help="SHA-256 identity of the persisted question bank",
    )
