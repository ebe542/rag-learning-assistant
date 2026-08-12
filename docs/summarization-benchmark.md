# Summarization benchmark

This document records manual performance measurements for document-wide summarization. The results establish reproducible local observations; they are not a portable performance guarantee because model runtime depends on the GPU, drivers, PyTorch, Transformers, generated output, and machine state.

## Environment

- Date: 12 August 2026
- GPU: NVIDIA GeForce RTX 3060 Ti with 8 GB VRAM
- Python: 3.13.14
- PyTorch: 2.13.0+cu132
- Generator: `Qwen/Qwen3-1.7B`, pinned project revision
- Source: `benchmarks/fixtures/summarization-document.pdf`
- Source SHA-256: `160df07726bf354eb6583baf94ead6e84814443e80a4c5e6b50c1ff3a1b47517`
- Source size: 10 pages and 20 indexed chunks
- Cache: bypassed with `--ignore-cache`

The first measured generation includes lazy model initialization. Model files had already been downloaded to the local Hugging Face cache. Each configuration was measured once, so repeated runs are required before treating small differences as significant.

## Reproducing the benchmark

Run the benchmark from the repository root:

```bash
rag-learn index benchmarks/fixtures/summarization-document.pdf --index-dir local-data/indexes/summarization-benchmark
python scripts/benchmark_summarization.py local-data/indexes/summarization-benchmark <DOCUMENT_UUID> --max-map-new-tokens 192 --max-reduce-new-tokens 384 --max-batch-chars 8000 --ignore-cache
```

`--ignore-cache` bypasses cache reads and writes for that invocation without deleting existing entries. Omit it when measuring normal resume behavior.

## Results

| Maximum batch characters | Maximum new tokens | Planned Map batches | Recorded generation | Total | Peak GPU | Result |
|---:|---:|---:|---:|---:|---:|---|
| 12,000 | 192 | 2 | unavailable | 20.12 s | 4,785.67 MB | First Map response and repair ended with invalid JSON |
| 8,000 | 192 | 2 | 17.88 s | 17.94 s | 4,173.00 MB | First Map response and repair ended with invalid JSON |

The first run preceded failed-call instrumentation, so it reported no individual generation measurement. The second run recorded one failed Map call with 8,457 input characters. `output_characters` is zero for a failed logical call because strict parsing did not produce a trusted `GenerationResult`; it does not mean the model emitted no raw text.

## Interpretation

Neither measured configuration completed its first Map batch. The model exhausted the shared 192-token output budget while producing JSON, and its single repair attempt also remained invalid. The measurements therefore describe failure behavior and must not be used to claim successful end-to-end throughput.

The `8,000 / 192` failure used less peak allocated GPU memory and ended sooner than `12,000 / 192`, but one run per configuration is insufficient to attribute those differences solely to batch size. The failure demonstrates that input batching and output control are separate concerns.

The subsequent implementation gave the Map prompt an explicit concise-output contract and introduced separate Map and Reduce token limits. The failed measurements above remain the evidence that motivated that change. A successful comparison can now repeat both character budgets against this unchanged fixture. Token-based input batching would also be more representative of model cost than character-based batching.

A later smoke run with the concise Map prompt showed that 128 Map tokens still
produced truncated JSON, while 192 Map tokens completed both partial summaries.
The first successful Reduce response copied schema placeholder text; removing the
copyable placeholder from the system prompt fixed that failure mode. A subsequent
response summarized only the first partial summary, so the application now requires
at least one supporting citation from every Map result and the Reduce prompt
explicitly requires document-wide coverage. Once validation was strengthened to
preserve the complete Map citation union, a 256-token Reduce run ended with
truncated JSON. Increasing only the Reduce budget to 384 produced valid JSON, a
document-wide synthesis, and all citations 1 through 20. This smoke result
supports the new default, but it is not a timing baseline because the captured
output did not include elapsed-time or per-call measurements.

Because these are single measurements and the first call includes initialization, no speedup is claimed. Future comparisons should record multiple successful runs, generated token counts, runtime versions, and summary-quality observations.
