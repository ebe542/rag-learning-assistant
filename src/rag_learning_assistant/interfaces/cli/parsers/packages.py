"""Register user-facing learning-package commands."""

from pathlib import Path

from rag_learning_assistant.interfaces.cli.parsing import (
    DEFAULT_QUESTION_COUNT,
    SubcommandCollection,
    non_blank_text,
    positive_int,
)


def add_package_commands(
    commands: SubcommandCollection,
) -> None:
    """Register product-level workflows over technical commands."""

    prepare_parser = commands.add_parser(
        "prepare",
        help="Turn one PDF into a ready-to-study learning package",
    )
    prepare_parser.add_argument(
        "pdf",
        type=Path,
        help="PDF document used to create the learning package",
    )
    prepare_parser.add_argument(
        "--library",
        type=Path,
        required=True,
        help="Directory containing the personal learning library",
    )
    prepare_parser.add_argument(
        "--name",
        type=non_blank_text,
        help="Optional package name; defaults to the PDF filename",
    )
    prepare_parser.add_argument(
        "--questions",
        dest="question_count",
        type=positive_int,
        default=DEFAULT_QUESTION_COUNT,
        help=(f"Number of study questions to prepare (default: {DEFAULT_QUESTION_COUNT})"),
    )
    list_parser = commands.add_parser(
        "package-list",
        help="List the learning packages in a personal library",
    )
    list_parser.add_argument(
        "--library",
        type=Path,
        required=True,
        help="Directory containing the personal learning library",
    )
    show_parser = commands.add_parser(
        "package-show",
        help="Show one learning package selected by name",
    )
    show_parser.add_argument(
        "--library",
        type=Path,
        required=True,
        help="Directory containing the personal learning library",
    )
    show_parser.add_argument(
        "--package",
        type=non_blank_text,
        required=True,
        help="Name of the learning package to show",
    )
    remove_parser = commands.add_parser(
        "package-remove",
        help="Remove one learning package and its learning data",
    )
    remove_parser.add_argument(
        "--library",
        type=Path,
        required=True,
        help="Directory containing the personal learning library",
    )
    remove_parser.add_argument(
        "--package",
        type=non_blank_text,
        required=True,
        help="Name of the learning package to remove",
    )
    progress_parser = commands.add_parser(
        "progress",
        help="Show learning progress for one ready package",
    )
    progress_parser.add_argument(
        "--library",
        type=Path,
        required=True,
        help="Directory containing the personal learning library",
    )
    progress_parser.add_argument(
        "--package",
        type=non_blank_text,
        required=True,
        help="Name of the learning package to report",
    )
