# RAG Learning Assistant

An open-source learning assistant that turns your PDF documents into source-grounded learning material, questions, and interactive tutoring sessions.

The project is an early local-first alpha. It can turn a text-based PDF into a
resumable learning package, conduct source-grounded written-answer sessions,
schedule reviews, and report learning progress. Its interfaces and stored-data
format may still change before a stable release.

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
- prepare resumable learning packages containing an indexed document, a
  grounded summary, and a grounded question bank

Scanned documents do not yet support OCR.

## Start here

New users should follow the concise
[alpha getting-started guide](docs/getting-started.md). It covers installation,
environment diagnosis, preparing the first PDF, studying, progress reporting,
and package removal with the user-facing commands.

## Product workflow

Turn one PDF into a ready-to-study package without copying document UUIDs or
generation fingerprints:

```bash
rag-learn prepare books/python-basics.pdf \
  --name "Python Basics" \
  --questions 20
```

The name defaults to the PDF filename when `--name` is omitted. Preparation
indexes the document, creates a persisted grounded summary, generates a
persisted question bank, and records a checkpoint after every expensive phase.
Repeating the command resumes an interrupted package or immediately reuses a
package that is already ready.

Question generation uses batches of three by default. Each validated batch is
stored in the library database before the next model call starts. Progress is
written as `Generating question batch N/M...`; a resumed run reports compatible
entries as `Using cached question batch N/M...`. Pressing `Ctrl+C` therefore
keeps completed batches, and repeating the same command continues with the
first missing batch. If a new batch repeats an earlier question, the application
keeps its unique candidates and generates each missing replacement in a separate
focused call with its own evidence subset. Accepted and rejected question texts
are marked as forbidden. Each replacement has at most three attempts, and every
failed text is added to the next attempt's forbidden list. Successive attempts
also change their required focus from a concrete example to a process and then
to a limitation or comparison. The requested question count is a target upper
bound: if no distinct replacement can be grounded after this limit, the
application persists the smaller valid bank and reports the shortfall instead
of discarding all generated learning material. After validation and
persistence, each newly generated batch reports its total elapsed time as
`Generated question batch N/M in S.S seconds.`; this duration includes a
possible replacement calls.

Three questions keep the default 512-token response budget reliable for concise
expected answers on the local model. Smaller batches require more calls but
reduce truncated JSON and narrow each call to fewer source contexts.

The persisted summary selects the trusted citations and remains part of the
question-bank identity, but its complete generated text is not sent to question
batches. The citations are distributed into stable, balanced evidence ranges,
and a question may cite only contexts assigned to its batch. This encourages
coverage of different document sections and prevents later batches from
repeatedly using the same easiest passages.

List packages without loading embedding or generation models:

```bash
rag-learn package-list
```

The product commands print concise, human-readable output by default. Add
`--json` to `package-list`, `package-show`, or `progress` when consuming their
stable machine-readable output from a script.

Show one package by its user-facing name:

```bash
rag-learn package-show \
  --package "Python Basics"
```

Remove a package together with its indexed document, generated learning
material, review schedule, and study history:

```bash
rag-learn package-remove \
  --package "Python Basics"
```

Product commands store learning data in the platform-specific user-data
directory by default. On Windows this is
`%LOCALAPPDATA%\rag-learning-assistant\library`. Pass `--library PATH` for a
different library, or set `RAG_LEARN_LIBRARY` to change the default for all
product commands.

Check whether Python, the local library, optional ML dependencies, and CUDA are
ready without downloading a model or changing application data:

```bash
rag-learn doctor
```

Add `--json` for a machine-readable support report. The command exits with
status 1 when the full local learning workflow needs attention.

UUIDs and SHA-256 identities remain in the JSON output for provenance and
automation, but they are not required for the product-level workflow.

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
python -m pip install -e ".[dev,local]"
```

### NVIDIA GPU support on Windows

The regular Python package index may install a CPU-only PyTorch build.
For an NVIDIA GPU, install the CUDA-enabled PyTorch build before installing the embedding dependencies.
The following setup was verified with Python 3.13, PyTorch 2.13.0, CUDA 13.2, and an NVIDIA GeForce RTX 3060 Ti:

```bash
py -3.13 -m venv .venv && source .venv/Scripts/activate
python -m pip install --upgrade pip && python -m pip install torch==2.13.0 --index-url https://download.pytorch.org/whl/cu132
python -m pip install -e ".[dev,local]"
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

Completed final summaries are also stored in `metadata.sqlite3`. Repeating the
command with the same document, model revision, prompts, and generation limits
returns the validated result without loading chunks or invoking the model. Use
`--force` to bypass both final and Map caches, regenerate every phase, and explicitly
replace the final result:

```bash
rag-learn summarize \
  local-data/indexes/learning \
  12345678-1234-5678-1234-567812345678 \
  --force
```

List the persisted generation identities for a document without loading the
generation model:

```bash
rag-learn summary-list \
  local-data/indexes/learning \
  12345678-1234-5678-1234-567812345678
```

Use an identity fingerprint from that output to retrieve the complete grounded
summary with citations and prompt references:

```bash
rag-learn summary-show \
  local-data/indexes/learning \
  12345678-1234-5678-1234-567812345678 \
  GENERATION_IDENTITY_SHA256
```

Removing a document also removes all of its persisted final summaries.
Replacing a document preserves its UUID but removes summaries derived from the
previous content after the new chunks have been indexed successfully.

Generate a persistent grounded question bank from one exact stored summary:

```bash
rag-learn question-generate \
  local-data/indexes/learning \
  12345678-1234-5678-1234-567812345678 \
  SUMMARY_IDENTITY_SHA256 \
  --count 5
```

The selected summary supplies the trusted source citations and exact generation
identity. Its generated overview text is intentionally omitted from question
prompts so batches remain focused on their assigned evidence. Repeating the same
configuration reuses the stored bank; add `--force` to regenerate and replace
it. List all banks for a document or retrieve one full bank without loading the
generation model:

Question-bank identity includes the batch size as well as the requested count,
model revision, prompt versions, token limit, and exact persisted summary.
Changing any of these inputs selects a different cache identity. `--force`
regenerates the complete bank without reading or writing intermediate batches.

```bash
rag-learn question-list \
  local-data/indexes/learning \
  12345678-1234-5678-1234-567812345678
```

```bash
rag-learn question-show \
  local-data/indexes/learning \
  12345678-1234-5678-1234-567812345678 \
  QUESTION_BANK_IDENTITY_SHA256
```

Removing or successfully replacing a document also removes question banks
derived from its previous content.

List the next due questions from one exact question bank:

```bash
rag-learn review-due \
  local-data/indexes/learning \
  12345678-1234-5678-1234-567812345678 \
  QUESTION_BANK_IDENTITY_SHA256 \
  --limit 10
```

Questions that have never been reviewed are due immediately. Existing scheduled
reviews are returned first by oldest due date, followed by new questions in bank
order. Record a self-rating after answering one question:

```bash
rag-learn review-record \
  local-data/indexes/learning \
  12345678-1234-5678-1234-567812345678 \
  QUESTION_BANK_IDENTITY_SHA256 \
  1 \
  good
```

Run one interactive active-recall session for the highest-priority due question
in a ready learning package:

```bash
rag-learn study \
  --package "RAG Learning Assistant"
```

The package name is the normal user-facing selector. The technical positional
form with library directory, document UUID, and question-bank fingerprint
remains available for diagnostics and automation that must address an exact
persisted question-bank version.

Inspect current progress for the same package without loading an ML model:

```bash
rag-learn progress \
  --package "RAG Learning Assistant"
```

The JSON report separates distinct questions from attempts because one question
may be answered repeatedly. It includes answered and due question counts,
verdict counts, answer and correctness rates, the latest study time, the next
due time, and concepts repeatedly reported as missing. Historical attempts that
predate automatic evaluation remain visible as `unclassified` instead of being
silently treated as correct or incorrect.

The expected answer and trusted sources remain hidden until a written answer has
been entered and evaluated by the local model. The model returns a validated
verdict, score, feedback, and missing concepts. The application deterministically
maps `incorrect` to `again`, `partially_correct` to `hard`, and `correct` to
`good`; the model never controls scheduling and one answer never produces
`easy`. Feedback, prompt provenance, the answer, question snapshot, citations,
and resulting progress are stored as one immutable attempt. Progress and attempt
writes share one SQLite transaction. Removing or successfully replacing the
document also removes its attempt history.

Supported ratings are `again`, `hard`, `good`, and `easy`. The command persists
the updated repetition count, ease factor, interval, and next due timestamp.
Removing or successfully replacing the source document also removes its review
progress.

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

Evaluate grounded question answering against the versioned benchmark cases:

```bash
python scripts/evaluate_grounded_generation.py \
  local-data/indexes/summarization-benchmark \
  --output local-data/evaluation/grounded-generation-baseline.json
```

The evaluation uses the production question-answering pipeline and reports exact
source-page citation coverage plus deterministic recall for curated key concepts.
Generated answer text is included for diagnosing failures. Accepted phrases make
the check reproducible but do not replace human factual review or prove semantic
equivalence.

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

The package-level classes, protocols, inheritance, and runtime relationships are
maintained as a renderable [PlantUML class overview](docs/class-overview.puml).

Planned next milestones:

1. add manual correction for incorrectly evaluated study answers
2. generate detailed section-level learning material
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
