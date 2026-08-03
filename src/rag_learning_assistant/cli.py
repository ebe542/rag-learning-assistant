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
    InMemoryVectorStore,
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

    search_parser = commands.add_parser(
        "search",
        help="Search a PDF semantically",
    )
    add_chunking_arguments(search_parser)
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


def build_document_search(
    chunker: TextChunker,
) -> DocumentSearchService:
    """Build the local semantic-search application service."""

    retrieval = RetrievalService(
        embedder=SentenceTransformerEmbedder(),
        store=InMemoryVectorStore(),
    )
    return DocumentSearchService(
        chunker=chunker,
        retrieval=retrieval,
    )


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
    document: Document,
    chunker: TextChunker,
    query: str,
    limit: int,
) -> int:
    """Index a document and write ranked search results as JSON."""

    search = build_document_search(chunker)
    search.index_document(document)
    results = search.search(query, limit=limit)

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

    return run_search(
        document=document,
        chunker=chunker,
        query=args.query,
        limit=args.limit,
    )


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
