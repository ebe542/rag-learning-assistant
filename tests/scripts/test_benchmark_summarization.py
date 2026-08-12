import pytest

from rag_learning_assistant.application.summarization import SUMMARY_REDUCE_PROMPT
from rag_learning_assistant.generation import GenerationResult
from scripts import benchmark_summarization


class StaticGenerator:
    def generate(self, prompt: str) -> GenerationResult:
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
    def generate(self, prompt: str) -> GenerationResult:
        raise ValueError("Invalid model response")


def test_timed_generator_records_map_and_reduce_calls(monkeypatch) -> None:
    times = iter((10.0, 12.5, 20.0, 21.0))
    monkeypatch.setattr(benchmark_summarization.torch.cuda, "synchronize", lambda: None)
    monkeypatch.setattr(benchmark_summarization.time, "perf_counter", lambda: next(times))
    generator = benchmark_summarization.TimedGenerator(StaticGenerator())

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
    monkeypatch.setattr(benchmark_summarization.torch.cuda, "synchronize", lambda: None)
    monkeypatch.setattr(benchmark_summarization.time, "perf_counter", lambda: next(times))
    generator = benchmark_summarization.TimedGenerator(FailingGenerator())

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
