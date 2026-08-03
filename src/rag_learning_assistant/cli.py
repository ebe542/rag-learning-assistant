"""Command-line entry point for the first ingestion milestone."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from rag_learning_assistant.ingestion import PdfExtractor


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Extract cited text from a PDF")
    parser.add_argument("pdf", type=Path, help="Path to a text-based PDF")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    document = PdfExtractor().extract(args.pdf)
    payload = {
        "source": document.source,
        "pages": [
            {"number": page.number, "source": page.source, "text": page.text}
            for page in document.pages
        ],
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
