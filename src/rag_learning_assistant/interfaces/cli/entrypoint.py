"""Command-line argument dispatch."""

from collections.abc import Sequence

from rag_learning_assistant.chunking import TextChunker
from rag_learning_assistant.ingestion import PdfExtractor
from rag_learning_assistant.interfaces.cli import commands
from rag_learning_assistant.interfaces.cli.parser import (
    build_parser,
    validate_existing_index_directory,
    validate_index_directory,
    validate_library_directory,
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

    if args.command == "list":
        try:
            validate_library_directory(args.index_dir)
        except ValueError as exc:
            parser.error(str(exc))

        return commands.run_list(args.index_dir)

    if args.command == "index":
        try:
            validate_index_directory(args.index_dir)
        except ValueError as exc:
            parser.error(str(exc))

    try:
        chunker = TextChunker(
            max_chars=args.max_chars,
            overlap_chars=args.overlap_chars,
        )
    except ValueError as exc:
        parser.error(str(exc))

    if args.command == "index":
        return commands.run_index(
            pdf_paths=args.pdfs,
            chunker=chunker,
            index_directory=args.index_dir,
        )

    document = PdfExtractor().extract(args.pdf)
    return commands.run_extract(document, chunker)
