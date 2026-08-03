"""Command-line entry point for the first ingestion milestone."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from rag_learning_assistant.chunking import TextChunker
from rag_learning_assistant.ingestion import PdfExtractor

DEFAULT_MAX_CHARS = 1000
DEFAULT_OVERLAP_CHARS = 150


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Extract cited text from a PDF")
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
    return parser


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
    chunks = chunker.chunk_pages(document.pages)
    payload = {
        "source": document.source,
        "pages": [
            {"number": page.number, "source": page.source, "text": page.text}
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


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
