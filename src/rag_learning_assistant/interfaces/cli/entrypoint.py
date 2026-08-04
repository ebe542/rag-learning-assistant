"""Command-line argument dispatch."""

from collections.abc import Sequence

from rag_learning_assistant.chunking import TextChunker
from rag_learning_assistant.ingestion import PdfExtractor
from rag_learning_assistant.interfaces.cli import commands
from rag_learning_assistant.interfaces.cli.parser import (
    build_parser,
    validate_empty_index_directory,
    validate_existing_index_directory,
)


def main(argv: Sequence[str] | None = None) -> int:
    """Parse arguments and execute the selected command."""

    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "search":
        try:
            validate_existing_index_directory(args.index_dir)
        except ValueError as exc:
            parser.error(str(exc))

        return commands.run_search(
            index_directory=args.index_dir,
            query=args.query,
            limit=args.limit,
        )

    if args.command == "index":
        try:
            validate_empty_index_directory(args.index_dir)
        except ValueError as exc:
            parser.error(str(exc))

    try:
        chunker = TextChunker(
            max_chars=args.max_chars,
            overlap_chars=args.overlap_chars,
        )
    except ValueError as exc:
        parser.error(str(exc))

    document = PdfExtractor().extract(args.pdf)

    if args.command == "extract":
        return commands.run_extract(document, chunker)

    return commands.run_index(
        document=document,
        chunker=chunker,
        index_directory=args.index_dir,
    )
