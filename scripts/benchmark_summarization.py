"""Manual GPU benchmark for document-wide summarization."""

import argparse
import json
import sys
import time
from contextlib import suppress
from dataclasses import asdict, dataclass
from pathlib import Path
from uuid import UUID

import torch
from dotenv import load_dotenv

from rag_learning_assistant.application.summarization import SUMMARY_REDUCE_PROMPT
from rag_learning_assistant.generation import GenerationResult, TextGenerator
from rag_learning_assistant.generation.cache import CachedSummaryBatch, SummaryBatchCache
from rag_learning_assistant.interfaces.cli.commands import (
    build_document_summarization_service,
)
from rag_learning_assistant.interfaces.cli.parser import (
    DEFAULT_SUMMARY_MAX_BATCH_CHARS,
    DEFAULT_SUMMARY_MAX_NEW_TOKENS,
    positive_int,
    validate_existing_index_directory,
)


@dataclass(frozen=True, slots=True)
class GenerationMeasurement:
    """Measurements for one real local-model invocation."""

    phase: str
    input_characters: int
    elapsed_seconds: float
    output_characters: int


class TimedGenerator:
    """Measure generation without changing the wrapped adapter's behavior."""

    def __init__(self, generator: TextGenerator) -> None:
        self.generator = generator
        self.measurements: list[GenerationMeasurement] = []

    def generate(self, prompt: str) -> GenerationResult:
        # Synchronizing prevents asynchronous CUDA work from escaping the timer.
        torch.cuda.synchronize()
        started_at = time.perf_counter()
        result = self.generator.generate(prompt)
        torch.cuda.synchronize()

        phase = "reduce" if prompt.startswith(SUMMARY_REDUCE_PROMPT.text) else "map"
        self.measurements.append(
            GenerationMeasurement(
                phase=phase,
                input_characters=len(prompt),
                elapsed_seconds=time.perf_counter() - started_at,
                output_characters=len(result.text),
            )
        )
        return result


class MeasuredSummaryCache:
    """Count cache decisions while preserving persistent-cache behavior."""

    def __init__(self, cache: SummaryBatchCache) -> None:
        self.cache = cache
        self.hits = 0
        self.misses = 0
        self.writes = 0

    def find_batch(
        self,
        identity_fingerprint: str,
        batch_number: int,
    ) -> CachedSummaryBatch | None:
        batch = self.cache.find_batch(identity_fingerprint, batch_number)
        if batch is None:
            self.misses += 1
        else:
            self.hits += 1
        return batch

    def save_batch(self, batch: CachedSummaryBatch) -> None:
        self.cache.save_batch(batch)
        self.writes += 1


def build_parser() -> argparse.ArgumentParser:
    """Build arguments for the manual benchmark."""

    parser = argparse.ArgumentParser(
        description="Benchmark real GPU-backed document summarization",
    )
    parser.add_argument("index_dir", type=Path)
    parser.add_argument("document_id", type=UUID)
    parser.add_argument(
        "--max-new-tokens",
        type=positive_int,
        default=DEFAULT_SUMMARY_MAX_NEW_TOKENS,
    )
    parser.add_argument(
        "--max-batch-chars",
        type=positive_int,
        default=DEFAULT_SUMMARY_MAX_BATCH_CHARS,
    )
    parser.add_argument(
        "--ignore-cache",
        action="store_true",
        help="Generate every map batch without reading or changing cached batches",
    )
    return parser


def main() -> int:
    """Run one benchmark and write machine-readable measurements as JSON."""

    project_root = Path(__file__).resolve().parents[1]
    with suppress(Exception):
        load_dotenv(dotenv_path=project_root / ".env")

    args = build_parser().parse_args()
    try:
        validate_existing_index_directory(args.index_dir)
    except ValueError as exc:
        print(f"Benchmark failed: {exc}", file=sys.stderr)
        return 1

    if not torch.cuda.is_available():
        print("Benchmark requires a CUDA-capable GPU.", file=sys.stderr)
        return 1

    service = build_document_summarization_service(
        index_directory=args.index_dir,
        max_new_tokens=args.max_new_tokens,
        max_batch_chars=args.max_batch_chars,
    )
    timed_generator = TimedGenerator(service.generator)
    service.generator = timed_generator

    measured_cache: MeasuredSummaryCache | None = None
    if args.ignore_cache:
        # Bypassing both paired dependencies preserves the service invariant and
        # leaves existing SQLite cache entries untouched.
        service.cache = None
        service.identity_factory = None
    elif service.cache is not None:
        measured_cache = MeasuredSummaryCache(service.cache)
        service.cache = measured_cache

    torch.cuda.reset_peak_memory_stats()
    started_at = time.perf_counter()

    try:
        summary = service.summarize(args.document_id)
        torch.cuda.synchronize()
    except KeyboardInterrupt:
        print("Summarization benchmark cancelled by user.", file=sys.stderr)
        return 130
    except Exception as exc:
        # Preserve completed measurements when grounding validation fails. This
        # makes a multi-minute failed run useful for performance comparison.
        with suppress(Exception):
            torch.cuda.synchronize()
        measurements = timed_generator.measurements
        elapsed_seconds = time.perf_counter() - started_at
        print(
            json.dumps(
                {
                    "status": "failed",
                    "error": str(exc),
                    "configuration": {
                        "max_new_tokens": args.max_new_tokens,
                        "max_batch_chars": args.max_batch_chars,
                        "cache_enabled": not args.ignore_cache,
                    },
                    "cache": {
                        "hits": measured_cache.hits if measured_cache is not None else 0,
                        "misses": measured_cache.misses if measured_cache is not None else 0,
                        "writes": measured_cache.writes if measured_cache is not None else 0,
                    },
                    "generation_calls": [asdict(item) for item in measurements],
                    "totals": {
                        "generation_calls": len(measurements),
                        "generation_seconds": sum(item.elapsed_seconds for item in measurements),
                        "elapsed_seconds": elapsed_seconds,
                    },
                    "gpu": {
                        "name": torch.cuda.get_device_name(),
                        "peak_memory_mb": torch.cuda.max_memory_allocated() / 1024**2,
                    },
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        print(f"Summarization benchmark failed: {exc}", file=sys.stderr)
        return 1

    total_seconds = time.perf_counter() - started_at
    measurements = timed_generator.measurements
    generated_seconds = sum(item.elapsed_seconds for item in measurements)

    payload = {
        "status": "passed",
        "document_id": str(summary.document_id),
        "source": summary.source,
        "configuration": {
            "max_new_tokens": args.max_new_tokens,
            "max_batch_chars": args.max_batch_chars,
            "cache_enabled": not args.ignore_cache,
        },
        "cache": {
            "hits": measured_cache.hits if measured_cache is not None else 0,
            "misses": measured_cache.misses if measured_cache is not None else 0,
            "writes": measured_cache.writes if measured_cache is not None else 0,
        },
        "generation_calls": [asdict(item) for item in measurements],
        "totals": {
            "generation_calls": len(measurements),
            "generation_seconds": generated_seconds,
            "elapsed_seconds": total_seconds,
            "summary_characters": len(summary.text),
            "citation_count": len(summary.citations),
        },
        "gpu": {
            "name": torch.cuda.get_device_name(),
            "peak_memory_mb": torch.cuda.max_memory_allocated() / 1024**2,
        },
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
