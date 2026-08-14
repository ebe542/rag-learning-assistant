"""Register document ingestion and library-management commands."""

from pathlib import Path
from uuid import UUID

from rag_learning_assistant.interfaces.cli.parsing import (
    SubcommandCollection,
    add_chunking_options,
)


def add_document_commands(commands: SubcommandCollection) -> None:
    """Register commands that create or mutate library documents."""

    extract_parser = commands.add_parser("extract", help="Extract pages and chunks as JSON")
    extract_parser.add_argument("pdf", type=Path, help="Path to a text-based PDF")
    add_chunking_options(extract_parser)

    index_parser = commands.add_parser("index", help="Add one or more PDFs to a persistent library")
    index_parser.add_argument(
        "pdfs", nargs="+", type=Path, help="Paths to text-based PDF documents"
    )
    add_chunking_options(index_parser)
    index_parser.add_argument(
        "--index-dir",
        type=Path,
        required=True,
        help="Directory for the FAISS index and SQLite metadata",
    )

    list_parser = commands.add_parser("list", help="List documents in a persistent library")
    list_parser.add_argument(
        "index_dir", type=Path, help="Directory containing the library metadata"
    )

    remove_parser = commands.add_parser(
        "remove", help="Remove a document from a persistent library"
    )
    remove_parser.add_argument("document_id", type=UUID, help="UUID of the document to remove")
    remove_parser.add_argument(
        "--index-dir",
        type=Path,
        required=True,
        help="Directory containing the library index",
    )

    replace_parser = commands.add_parser(
        "replace", help="Replace a library document while preserving its UUID"
    )
    replace_parser.add_argument("document_id", type=UUID, help="UUID of the document to replace")
    replace_parser.add_argument("pdf", type=Path, help="Path to the replacement PDF")
    replace_parser.add_argument(
        "--index-dir",
        type=Path,
        required=True,
        help="Directory containing the library index",
    )
    add_chunking_options(replace_parser)
