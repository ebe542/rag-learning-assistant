# Benchmark fixtures

This directory contains stable inputs for manual performance measurements. All fixture content is original project material distributed under the repository's MIT license. It contains no private documents or third-party book text.

## Summarization fixture

`fixtures/summarization-document.pdf` is a ten-page synthetic learning document about building a source-grounded assistant. Its editable source is `fixtures/summarization-source.md`.

Expected PDF SHA-256:

```text
160df07726bf354eb6583baf94ead6e84814443e80a4c5e6b50c1ff3a1b47517
```

Regenerate the PDF from the repository root:

```bash
python benchmarks/generate_fixture.py
```

The generator uses only the project's existing PyMuPDF dependency. It fixes page dimensions, typography, metadata, and PDF identifiers so unchanged source produces stable bytes. Commit both source and PDF whenever the fixture content intentionally changes.

Index the fixture in a dedicated local library before benchmarking:

```bash
rag-learn index benchmarks/fixtures/summarization-document.pdf --index-dir local-data/indexes/summarization-benchmark
```

Use the document UUID returned by that command with `scripts/benchmark_summarization.py`. The generated index remains local and must not be committed.
