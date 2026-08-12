import pytest

from rag_learning_assistant.application.summarization import SUMMARY_REDUCE_PROMPT
from rag_learning_assistant.generation import GenerationResult
from scripts import benchmark_summarization


class StaticGenerator:
    def generate(
        self,
        prompt: str,
        *,
        max_new_tokens: int | None = None,
    ) -> GenerationResult:
        return GenerationResult(text="Summary", citation_numbers=(1,))


class EmptyCache:
    def find_batch(
        self,
        identity_fingerprint: str,
        batch_number: int,
    ) -> None:
        return None

    def save_batch(self, batch: object) -> None:
        return None


class FailingGenerator:
    def generate(
        self,
        prompt: str,
        *,
        max_new_tokens: int | None = None,
    ) -> GenerationResult:
        raise ValueError("Invalid model response")


class RecordingCuda:
    """Keep timing tests independent from the optional Torch installation."""

    def synchronize(self) -> None:
        return None

    def is_available(self) -> bool:
        return True

    def reset_peak_memory_stats(self) -> None:
        return None

    def get_device_name(self) -> str:
        return "Test GPU"

    def max_memory_allocated(self) -> int:
        return 0


def test_timed_generator_records_map_and_reduce_calls(monkeypatch) -> None:
    times = iter((10.0, 12.5, 20.0, 21.0))
    monkeypatch.setattr(benchmark_summarization.time, "perf_counter", lambda: next(times))
    generator = benchmark_summarization.TimedGenerator(StaticGenerator(), RecordingCuda())

    generator.generate("Map prompt")
    generator.generate(f"{SUMMARY_REDUCE_PROMPT.text}\n\nSections")

    assert [measurement.phase for measurement in generator.measurements] == [
        "map",
        "reduce",
    ]
    assert [measurement.elapsed_seconds for measurement in generator.measurements] == [
        2.5,
        1.0,
    ]
    assert [measurement.status for measurement in generator.measurements] == [
        "passed",
        "passed",
    ]


def test_timed_generator_records_failed_calls(monkeypatch) -> None:
    times = iter((10.0, 14.0))
    monkeypatch.setattr(benchmark_summarization.time, "perf_counter", lambda: next(times))
    generator = benchmark_summarization.TimedGenerator(FailingGenerator(), RecordingCuda())

    with pytest.raises(ValueError, match="Invalid model response"):
        generator.generate("Map prompt")

    assert generator.measurements == [
        benchmark_summarization.GenerationMeasurement(
            phase="map",
            status="failed",
            input_characters=10,
            elapsed_seconds=4.0,
            output_characters=0,
            error="Invalid model response",
        )
    ]


def test_measured_cache_counts_misses_and_writes() -> None:
    cache = benchmark_summarization.MeasuredSummaryCache(EmptyCache())

    assert cache.find_batch("a" * 64, 1) is None

    assert cache.hits == 0
    assert cache.misses == 1
    assert cache.writes == 0


def test_parser_accepts_separate_map_and_reduce_token_limits() -> None:
    args = benchmark_summarization.build_parser().parse_args(
        [
            "local-data/indexes/summarization-benchmark",
            "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa",
            "--max-map-new-tokens",
            "128",
            "--max-reduce-new-tokens",
            "256",
            "--max-batch-chars",
            "8000",
        ]
    )

    assert args.max_map_new_tokens == 128
    assert args.max_reduce_new_tokens == 256
    assert args.max_batch_chars == 8000
