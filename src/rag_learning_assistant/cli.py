"""Command-line entry point for the first ingestion milestone."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from rag_learning_assistant.application import DocumentSearchService
from rag_learning_assistant.chunking import TextChunker
from rag_learning_assistant.ingestion import Document, PdfExtractor
from rag_learning_assistant.retrieval import (
    FaissVectorStore,
    RetrievalService,
    SentenceTransformerEmbedder,
)

DEFAULT_MAX_CHARS = 1000
DEFAULT_OVERLAP_CHARS = 150
DEFAULT_RESULT_LIMIT = 5


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


def validate_empty_index_directory(index_directory: Path) -> None:
    """Require a new or empty directory for document indexing."""

    if not index_directory.exists():
        return

    if not index_directory.is_dir() or any(index_directory.iterdir()):
        raise ValueError("index directory must be empty")


def validate_existing_index_directory(index_directory: Path) -> None:
    """Require both persistent index files before starting retrieval."""

    required_files = (
        index_directory / "vectors.faiss",
        index_directory / "metadata.sqlite3",
    )

    if not index_directory.is_dir() or not all(path.is_file() for path in required_files):
        raise ValueError("index directory is incomplete")


def add_chunking_arguments(parser: argparse.ArgumentParser) -> None:
    """Add options shared by commands that process documents."""

    parser.add_argument("pdf", type=Path, help="Path to a text-based PDF")
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


def build_persistent_retrieval(
    index_directory: Path,
) -> RetrievalService:
    """Build retrieval backed by an existing persistent index."""

    embedder = SentenceTransformerEmbedder()
    store = FaissVectorStore(
        index_directory,
        model_name=embedder.model_name,
        model_revision=embedder.model_revision,
    )
    return RetrievalService(
        embedder=embedder,
        store=store,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build learning material from PDF documents")
    commands = parser.add_subparsers(
        dest="command",
        required=True,
    )

    extract_parser = commands.add_parser(
        "extract",
        help="Extract pages and chunks as JSON",
    )
    add_chunking_arguments(extract_parser)

    index_parser = commands.add_parser(
        "index",
        help="Create a persistent search index for a PDF",
    )
    add_chunking_arguments(index_parser)

    index_parser.add_argument(
        "--index-dir",
        type=Path,
        required=True,
        help="Directory for the FAISS index and SQLite metadata",
    )

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

    return parser


def build_persistent_document_search(
    chunker: TextChunker,
    index_directory: Path,
) -> DocumentSearchService:
    """Build document indexing backed by a persistent FAISS index."""

    return DocumentSearchService(
        chunker=chunker,
        retrieval=build_persistent_retrieval(index_directory),
    )


def run_index(
    document: Document,
    chunker: TextChunker,
    index_directory: Path,
) -> int:
    """Persist the searchable chunks of a document."""

    search = build_persistent_document_search(
        chunker,
        index_directory,
    )
    chunks = search.index_document(document)

    payload = {
        "source": document.source,
        "index_directory": str(index_directory),
        "chunks_indexed": len(chunks),
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


def run_extract(
    document: Document,
    chunker: TextChunker,
) -> int:
    """Write extracted pages and chunks as JSON."""

    chunks = chunker.chunk_pages(document.pages)
    payload = {
        "source": document.source,
        "pages": [
            {
                "number": page.number,
                "source": page.source,
                "text": page.text,
            }
            for page in document.pages
        ],
        "chunks": [
            {
                "index": chunk.index,
                "text": chunk.text,
                "source": chunk.source,
                "page_number": chunk.page_number,
            }
            for chunk in chunks
        ],
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


def run_search(
    index_directory: Path,
    query: str,
    limit: int,
) -> int:
    """Search an existing index and write ranked results as JSON."""

    retrieval = build_persistent_retrieval(index_directory)
    results = retrieval.search(query, limit=limit)

    payload = {
        "query": query,
        "results": [
            {
                "score": result.score,
                "text": result.chunk.text,
                "source": result.chunk.source,
                "page_number": result.chunk.page_number,
                "index": result.chunk.index,
            }
            for result in results
        ],
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "search":
        try:
            validate_existing_index_directory(args.index_dir)
        except ValueError as exc:
            parser.error(str(exc))

        return run_search(
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
        return run_extract(document, chunker)

    return run_index(
        document=document,
        chunker=chunker,
        index_directory=args.index_dir,
    )


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
