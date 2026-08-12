# RAG Learning Assistant

An open-source learning assistant that turns your PDF documents into source-grounded learning material, questions, and interactive tutoring sessions.

The project is at an early MVP stage.
It extracts text page by page, preserves citation metadata, creates configurable paragraph-aware chunks, and provides the foundations for local semantic retrieval.
Question generation and learning-progress tracking will build on this foundation.

## Current features

- extract text from text-based PDF files
- retain document name and one-based page numbers
- output structured JSON through a small command-line interface
- testable core with no model or cloud-service dependency
- split pages into size-limited, overlapping chunks while preserving paragraph boundaries
- persist and search embeddings with FAISS and SQLite
- manage multiple documents in one local library
- detect duplicate document content using SHA-256
- create local multilingual E5 embeddings through an optional Sentence Transformers adapter
- build source-grounded answers through an optional local Qwen3 generator

Scanned documents do not yet support OCR.

## Quick start

Python 3.11 or newer is supported.
Python 3.13 is the recommended version for local ML development.

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
pytest
```

Install the optional local embedding, generation, and persistent-storage support:

```powershell
python -m pip install -e ".[dev,embeddings,generation,storage]"
```

### NVIDIA GPU support on Windows

The regular Python package index may install a CPU-only PyTorch build.
For an NVIDIA GPU, install the CUDA-enabled PyTorch build before installing the embedding dependencies.
The following setup was verified with Python 3.13, PyTorch 2.13.0, CUDA 13.2, and an NVIDIA GeForce RTX 3060 Ti:

```bash
py -3.13 -m venv .venv && source .venv/Scripts/activate
python -m pip install --upgrade pip && python -m pip install torch==2.13.0 --index-url https://download.pytorch.org/whl/cu132
python -m pip install -e ".[dev,embeddings,generation,storage]"
```

Verify that PyTorch can use the GPU:

```bash
python -c "import torch; print('Torch:', torch.__version__); print('CUDA:', torch.version.cuda); print('Available:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none')"
```

The CUDA wheel must match the installed NVIDIA driver and a version offered by PyTorch.
Refer to the [official PyTorch installation instructions](https://pytorch.org/get-started/locally/) when changing PyTorch or CUDA versions.

The default embedding model is `intfloat/multilingual-e5-small`, pinned to a specific Hugging Face revision for reproducibility.
The first real embedding request downloads the model into the local Hugging Face cache.
The first local generation provider uses `Qwen/Qwen3-1.7B`, also pinned to a specific revision.
It requests strict JSON containing answer text and references to numbered retrieval contexts; application code resolves those numbers to trusted source metadata.
A repository-root `.env` is loaded optionally by the CLI, allowing `HF_TOKEN` to be used without making authentication a requirement for public models.
A Windows CUDA smoke test used approximately 4.2 GB of GPU memory on an RTX 3060 Ti.

Extract pages and chunks from a PDF:

```powershell
rag-learn extract path\to\book.pdf --max-chars 1000 --overlap-chars 150
```

The command emits JSON containing the extracted pages and searchable chunks.
Every chunk retains its source file, page number, and document-wide index.
Chunk size and overlap can be configured through the command-line options shown above.

Create a persistent library and add one or more documents:

```bash
mkdir -p local-data/documents local-data/indexes
rag-learn index \
  local-data/documents/book.pdf \
  local-data/documents/notes.pdf \
  --index-dir local-data/indexes/learning
```

Add more documents to the same library and list its contents:

```bash
rag-learn index local-data/documents/notes.pdf --index-dir local-data/indexes/learning
rag-learn list local-data/indexes/learning
```

Replace a document while preserving the UUID shown by `list`:

```bash
rag-learn replace \
  12345678-1234-5678-1234-567812345678 \
  local-data/documents/revised-book.pdf \
  --index-dir local-data/indexes/learning
```

Remove a document by the stable UUID shown by `list`:

```bash
rag-learn remove \
  12345678-1234-5678-1234-567812345678 \
  --index-dir local-data/indexes/learning
```

Search the existing library without reopening or re-embedding its PDFs:

```bash
rag-learn search local-data/indexes/learning "What are Python functions?" --limit 3
```

Ask a source-grounded question using the retrieved passages and local generator:

```bash
rag-learn ask local-data/indexes/learning "What are Python functions?" --limit 3
```

The command emits JSON containing the answer, trusted citations with context number, source, page, chunk index, and excerpt, and compact references to the exact prompt versions used.
Grounding instructions require every factual claim to be supported by retrieved context and forbid the model from using prior knowledge.
The command is intended for focused questions; document-wide summaries require a separate summarization workflow because a small top-k result cannot represent an entire library reliably.

Summarize a complete indexed document using the UUID shown by `list`:

```bash
rag-learn summarize \
  local-data/indexes/learning \
  12345678-1234-5678-1234-567812345678
```

Completed map batches are cached in the library's `metadata.sqlite3` and reused
after an interrupted run when the document and generation configuration remain
unchanged.

Measure real local-model generation and cache behavior manually:

```bash
python scripts/benchmark_summarization.py \
  local-data/indexes/summarization-benchmark \
  DOCUMENT_UUID \
  --max-map-new-tokens 192 \
  --max-reduce-new-tokens 384 \
  --max-batch-chars 8000 \
  --ignore-cache
```

The benchmark reports each map and reduce call with input and output character
counts and elapsed time, plus cache counters, total runtime, and peak GPU memory.
The first generation measurement includes lazy model loading. Omit
`--ignore-cache` to measure normal resume behavior; using it bypasses cache reads
and writes without deleting existing entries. This GPU benchmark is intentionally
excluded from the automated test suite and CI.
The reproducible fixture, recorded RTX 3060 Ti measurements, and their limitations are documented in the
[summarization benchmark](docs/summarization-benchmark.md).

Each library directory contains `vectors.faiss` and `metadata.sqlite3`.
SQLite records documents with stable UUIDs and content hashes, maps FAISS IDs back to their document chunks, and stores embedding-model metadata.
Adding identical file content again is rejected before PDF extraction or embedding generation.
Batch imports process documents sequentially and report each path as `added`, `skipped`, or `failed`.
Duplicates are skipped, while other per-file errors do not prevent later inputs from being processed.
The command exits with status 1 when at least one document fails.
Removing a document deletes only its FAISS vectors, chunk metadata, and catalog entry; other library documents remain searchable.
Replacing validates, extracts, chunks, and embeds the new PDF before changing the stored document, then updates its vectors and metadata without changing its UUID.

## Planned architecture

```text
PDF -> page extraction -> semantic chunks -> embeddings -> vector search
                                                        -> grounded LLM tutor
                                                        -> learning materials
                                                        -> questions and progress
```

Planned next milestones:

1. generate summaries and reusable question banks
2. track learner feedback and spaced repetition
3. add optional Ollama and API model providers

## Development

Run the checks locally:

```bash
python scripts/check_milestone.py
```

The milestone check verifies formatting, lint rules, the complete test suite,
at least 90% coverage, unclosed resources, and invalid Git whitespace.

Before pushing a milestone, reproduce one GitHub CI job in a clean environment:

```bash
python scripts/check_ci_environment.py
```

This slower check creates a disposable virtual environment, installs exactly
the `dev,storage` extras used by CI, verifies that tooling remains importable
without the optional Torch generation stack, and runs the shared milestone
check. It uses the active Python version by default. To check another installed
matrix version, select its interpreter explicitly, for example:

```bash
python scripts/check_ci_environment.py --python python3.12
```

Contributions are welcome.
Please read [CONTRIBUTING.md](CONTRIBUTING.md).
For architecture and extension guidance, see the [development guide](docs/development.md) and [architecture decisions](docs/adr/README.md).

## License

This project is available under the [MIT License](LICENSE).
