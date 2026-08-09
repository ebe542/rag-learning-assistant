"""Manual GPU smoke test for document-wide summarization."""

import argparse
import sys
import time
from contextlib import suppress
from pathlib import Path
from uuid import UUID

import torch
from dotenv import load_dotenv

from rag_learning_assistant.interfaces.cli.commands import run_summarize
from rag_learning_assistant.interfaces.cli.parser import (
    DEFAULT_SUMMARY_MAX_BATCH_CHARS,
    DEFAULT_SUMMARY_MAX_NEW_TOKENS,
    positive_int,
    validate_existing_index_directory,
)


def build_parser() -> argparse.ArgumentParser:
    """Build arguments for the manual smoke test."""

    parser = argparse.ArgumentParser(
        description=(
            "Summarize one document from a real persistent index using the local Hugging Face model"
        )
    )
    parser.add_argument(
        "index_dir",
        type=Path,
        help="Directory containing the persistent library index",
    )
    parser.add_argument(
        "document_id",
        type=UUID,
        help="UUID of the document to summarize",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=positive_int,
        default=DEFAULT_SUMMARY_MAX_NEW_TOKENS,
        help=(
            "Maximum number of generated summary tokens "
            f"(default: {DEFAULT_SUMMARY_MAX_NEW_TOKENS})"
        ),
    )
    parser.add_argument(
        "--max-batch-chars",
        type=positive_int,
        default=DEFAULT_SUMMARY_MAX_BATCH_CHARS,
        help=(
            "Maximum source characters per summary batch "
            f"(default: {DEFAULT_SUMMARY_MAX_BATCH_CHARS})"
        ),
    )
    return parser


def main() -> int:
    """Run one real document summary and report peak GPU memory."""

    project_root = Path(__file__).resolve().parents[1]

    # Authentication improves Hub rate limits but remains optional because the
    # pinned embedding and generation models are publicly accessible.
    with suppress(Exception):
        load_dotenv(dotenv_path=project_root / ".env")

    args = build_parser().parse_args()

    try:
        validate_existing_index_directory(args.index_dir)
    except ValueError as exc:
        print(
            f"Smoke test failed: {exc}",
            file=sys.stderr,
        )
        return 1

    if not torch.cuda.is_available():
        print(
            "Smoke test requires a CUDA-capable GPU.",
            file=sys.stderr,
        )
        return 1

    # Include lazy model loading and generation in both measurements.
    torch.cuda.reset_peak_memory_stats()
    started_at = time.perf_counter()

    try:
        exit_code = run_summarize(
            index_directory=args.index_dir,
            document_id=args.document_id,
            max_new_tokens=args.max_new_tokens,
            max_batch_chars=args.max_batch_chars,
        )
    except KeyboardInterrupt:
        print(
            "Summarization smoke test cancelled by user.",
            file=sys.stderr,
        )
        return 130
    except Exception as exc:
        print(
            f"Summarization smoke test failed: {exc}",
            file=sys.stderr,
        )
        return 1
    finally:
        elapsed_seconds = time.perf_counter() - started_at
        print(
            f"Elapsed: {elapsed_seconds:.1f} seconds ({elapsed_seconds / 60:.2f} minutes)",
            file=sys.stderr,
        )

    print(
        f"GPU: {torch.cuda.get_device_name()}",
        file=sys.stderr,
    )
    print(
        f"Peak GPU MB: {torch.cuda.max_memory_allocated() / 1024**2:.1f}",
        file=sys.stderr,
    )
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
