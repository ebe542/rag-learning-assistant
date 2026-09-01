# Development Guide

This guide describes the current architecture, development workflow, persistence
rules, and extension points of the RAG Learning Assistant. For user-facing setup
and commands, see the project [README](../README.md). Long-lived design reasons
are recorded in the [Architecture Decision Records](adr/README.md).

## Current system

The project is a local, source-grounded learning system built around a persistent
document library:

```text
PDF
  -> page extraction
  -> page-bounded chunks
  -> E5 embeddings
  -> FAISS retrieval
  -> grounded answers
  -> persisted document summaries
  -> persisted grounded question banks
  -> persisted spaced-review schedules
  -> interactive study attempts
  -> user-facing resumable learning packages
```

The detailed classes, protocols, inheritance, and runtime relationships are
maintained in a renderable [PlantUML overview](class-overview.puml) and focused
[class diagrams](diagrams/README.md).

## Source responsibilities

```text
src/rag_learning_assistant/
├── application/
│   ├── batch_import.py
│   ├── answer_evaluation.py
│   ├── document_search.py
│   ├── library.py
│   ├── learning_package.py
│   ├── question_answering.py
│   ├── question_bank.py
│   ├── review.py
│   ├── study_session.py
│   ├── summarization.py
│   └── summary_catalog.py
├── chunking/
├── evaluation/
├── generation/
│   ├── question_cache.py
│   └── ...
├── ingestion/
├── interfaces/
│   └── cli/
│       ├── commands.py
│       ├── entrypoint.py
│       ├── parser.py
│       ├── parsing.py
│       └── parsers/
├── learning/
│   ├── package_repository.py
│   └── packages.py
├── library/
├── retrieval/
└── cli.py
```

The packages follow responsibility boundaries rather than technical convenience:

- `ingestion` reads source documents and preserves source metadata;
- `chunking` creates searchable, page-bounded text units;
- `retrieval` embeds, stores, and searches chunks;
- `library` owns persistent document metadata;
- `generation` owns model adapters, prompts, parsing, identities, and generation
  persistence;
- `learning` owns grounded question-bank, review-progress, immutable study-
  attempt, and user-facing learning-package domain models plus SQLite
  repositories;
- `application` coordinates use cases through narrow protocols;
- `evaluation` measures deterministic citation and concept coverage;
- `interfaces.cli` validates CLI input, wires concrete adapters, and serializes
  stable JSON output.

## Architectural conventions

Domain models are generally frozen, slotted dataclasses. They validate their own
state in `__post_init__`, while application services validate transitions that
require knowledge of previous state or external data.

Application services depend on `Protocol` interfaces. Concrete adapters satisfy
these protocols structurally and do not need to inherit from them. This keeps
the domain independent of PyMuPDF, FAISS, SQLite, Sentence Transformers, and
Transformers.

Feature packages export supported public names through `__init__.py`. Internal
modules may be imported directly when a concrete implementation is explicitly
required, but callers should prefer package-level imports.

The project uses stable identities at every persistent boundary:

- document UUID identifies a library entry across replacement;
- document SHA-256 identifies exact file content;
- model name and pinned revision identify embeddings and generation models;
- prompt name, version, and SHA-256 identify exact prompt text;
- generation fingerprints identify complete summary configurations;
- question-bank fingerprints include their selected persisted summary identity;
- review progress uses document UUID, question-bank fingerprint, and question
  number as its composite identity.
- study attempts use their own UUID while retaining the complete question-bank
  identity and resulting progress snapshot.
- learning packages use a UUID internally and a case-insensitive unique name as
  the user-facing selector for active document, summary, and question-bank
  identities.

## Processing stages

### Ingestion

`PdfExtractor` uses PyMuPDF to read text-based PDFs and returns immutable
`Document` and `Page` models. Page numbers are one-based and every page retains
the source filename. PDF handles are closed through their context-manager
contract. The extractor rejects password-protected files and files without
pages before page access, reports the one-based page number for a failed text
stream, removes invalid control characters, and requires at least one
machine-readable word across the document. Textless pages inside an otherwise
readable document remain present with their original page number; this boundary
allows a future OCR adapter to process only affected pages. OCR is intentionally
not part of the current ingestion adapter.

### Chunking

`TextChunker` creates immutable `Chunk` objects without crossing page boundaries.
Chunk indices increase across the complete document. The algorithm prefers
paragraph and word boundaries, hard-splits text only when necessary, and applies
overlap only to split paragraphs. Character-based limits keep the core
independent of a specific tokenizer.

### Embeddings and retrieval

`Embedder` separates document and query embeddings because asymmetric retrieval
models may encode them differently. `SentenceTransformerEmbedder` uses:

```text
Model:     intfloat/multilingual-e5-small
Revision:  614241f622f53c4eeff9890bdc4f31cfecc418b3
Dimension: 384
```

The adapter applies the E5 `passage: ` and `query: ` prefixes, normalizes output,
and loads the model lazily.

`InMemoryVectorStore` is the small reference implementation.
`FaissVectorStore` persists normalized vectors in an exact inner-product index
and stores chunk metadata, document IDs, and embedding identity in SQLite. For
normalized vectors, inner product is equivalent to cosine similarity.

FAISS and SQLite are separate stores because they serve different purposes:
FAISS ranks vectors efficiently, while SQLite preserves inspectable metadata and
maps FAISS IDs back to source chunks. Mutation methods validate complete batches
before replacing persisted state and preserve unrelated documents.

### Document library

`LibraryService` hashes content, rejects duplicates, assigns UUIDs, coordinates
indexing, and writes catalog metadata. `BatchImportService` processes paths
sequentially to avoid loading multiple GPU-backed embedding models and isolates
ordinary per-file failures.

Document replacement preserves the UUID but changes source metadata, content
hash, chunks, and vectors. Document removal verifies the removed chunk count
before deleting catalog metadata.

Summaries, question banks, review progress, and study attempts are derived data. After a
successful removal or replacement, `LibraryService` invokes every registered
`DocumentDerivedDataCleaner` before changing document metadata. If vector
mutation fails or reports inconsistent state, derived data and catalog metadata
remain available for diagnosis.

### Grounded question answering

`QuestionAnsweringService` retrieves a small top-k context set, builds a versioned
prompt, asks a `TextGenerator` for strict JSON, and maps model citation numbers
back to trusted retrieval results. Source metadata never comes from generated
text. The service rejects citation numbers that do not exist in the supplied
contexts.

Focused questions belong to this retrieval path. Document-wide requests do not:
a small similarity result cannot reliably cover an entire book.

### Local generation

`HuggingFaceTextGenerator` uses the pinned local model:

```text
Model:    Qwen/Qwen3-1.7B
Revision: 70d244cc86ccca08cf5af4e1e306ecf908b1ad5e
```

The adapter loads its pipeline lazily, uses deterministic generation, disables
Qwen reasoning output for structured responses, and parses strict JSON. An
invalid response receives one format-only repair attempt with an adaptively
larger output budget. The first repair receives at least 512 generated tokens
and normally doubles the caller's budget. If that response is still invalid and
its JSON is recognizably truncated, one final repair doubles the budget again,
up to the default 1,024-token ceiling. A non-truncation error, an exhausted
ceiling, or three unsuccessful total generation attempts fails immediately.
The ceiling and attempt count are configurable on the adapter, and a caller's
larger initial budget is never reduced. This bounded policy applies uniformly
to summaries, question batches, and answer evaluations. Repair does not add or
remove source facts or citation numbers. Every prompt has an explicit version
and SHA-256 reference; results report which prompts were actually used.

### Document-wide summarization

`DocumentSummarizationService` reads every stored chunk for one document and uses
a map-reduce workflow:

1. map batches summarize bounded source contexts;
2. successful map results are cached by generation identity and batch number;
3. reduce combines all partial summaries while preserving trusted original
   citation numbers;
4. the validated final summary is persisted by its exact generation identity.

Map and reduce token limits are separate because partial and final outputs have
different requirements. A normal rerun reuses compatible map batches and an
existing final summary. `--force` bypasses those reads, regenerates all phases,
and explicitly replaces the final result.

`DocumentSummaryCatalog` lists stored summary identities and retrieves one exact
grounded result without loading the model.

### Grounded question banks

`QuestionBankService` generates a requested number of free-response questions
from one explicitly selected persisted summary. The summary identity selects
the exact source state, while its stored citations are the only permitted
evidence. The generated summary text is deliberately omitted from question
prompts because a small deterministic model otherwise repeats global topics
across batches. Model-returned citation numbers are resolved back to trusted
`Citation` objects.

A question-bank identity includes model revision, all available prompt versions,
question count, batch size, token limit, and source-summary identity. Complete
banks are persisted in SQLite. `QuestionBankCatalog` lists and retrieves them
without loading generation dependencies.

Question generation uses batches of three by default. `QuestionBankService`
translates each independently parsed response into global question numbers and
persists the validated result through `QuestionBatchCache` before starting the
next model call. A compatible retry reuses cached batches by identity and batch
number. Citation, prompt-provenance, numbering, and cross-batch uniqueness checks
run before a new batch is saved. Earlier question texts are included in later
prompts to discourage repetition. If normalized question text is nevertheless
duplicated, the service retains the current batch's unique candidates and
generates every missing replacement in a separate semantic repair call. Each
call receives a distinct, balanced subset of the batch evidence. Its versioned
prompt explicitly forbids accepted earlier texts, accepted candidates, rejected
duplicates, and replacements already accepted in the same repair sequence. Each
replacement has at most three attempts. An exact duplicate from a failed attempt
is added to the following prompt's forbidden list. Because generation is
deterministic, the attempts also use different focus instructions: concrete
detail, process or cause, then limitation or comparison. Each repaired result
passes the same citation, provenance, and uniqueness checks. The requested count
is a target upper bound. If bounded repair cannot produce another distinct
question, the incomplete batch is not cached, the smaller valid final bank is
persisted, and a `shortfall` progress event reports actual and requested counts.

The default of three balances resumability against the local model's output
budget. Expected answers are limited by prompt to two sentences so one batch can
normally finish within 512 new tokens. Larger batches had repeatedly truncated
the fifth question in real German-language document runs.

The service deterministically partitions the persisted summary's trusted
citations into balanced, contiguous evidence ranges for the configured batch
plan. Each prompt contains only its batch's assigned contexts. Both initial and
replacement questions may cite only that range. When a request has more batches
than citations, citations are reused cyclically so no batch is generated without
evidence. The prompt version records this changed generation strategy and
therefore separates its cache identity from older all-context or full-summary
batches.

The optional progress callback distinguishes `generate` from `cached` batches.
The CLI writes these messages to standard error and flushes before expensive
model calls. A `completed` event reports monotonic elapsed time only after a new
batch has passed validation and optional persistence. Its duration covers the
complete batch operation, including duplicate-replacement calls when required.
Cached batches do not report an artificial duration. `force=True` bypasses
intermediate cache reads and writes, while a normal interruption leaves
completed batches available for the next run.

Terminal JSON-repair failures attach bounded diagnostics to the raised
exception and record up to 1,000 characters from each model response. The
central exception logger preserves these notes while the
console remains concise. Because the rotating per-user log may therefore contain
source-derived text, it must be treated as private diagnostic data and must not
be committed or shared without review.

Question banks distribute source evidence to cover multiple document sections.
Dedicated section-level learning units remain a separate future capability.

### Spaced review

`ReviewScheduler` is a deterministic, inspectable policy inspired by SM-2 but is
not presented as an exact implementation:

- `again` retries after ten minutes and resets successful repetitions;
- `hard` grows the interval conservatively and lowers ease;
- `good` uses one day, then six days, then multiplies by ease;
- `easy` starts at four days and grows faster;
- ease never drops below 1.3.

`ReviewService` validates the exact bank and question, loads current progress,
applies the scheduler, and saves the new immutable state. New questions have no
database row until their first recorded review. Due selection prioritizes the
oldest scheduled reviews before new questions, preventing old work from being
starved by newly generated material. Timestamps are timezone-aware and CLI events
use UTC.

`StudySessionService` selects one due question and requires a written learner
answer. `AnswerEvaluationService` compares it only with the expected answer and
trusted citation excerpts, treating every supplied field as untrusted data. The
validated result stores `incorrect`, `partially_correct`, or `correct`, a score,
constructive feedback, missing concepts, and complete prompt provenance.

The application deterministically maps those verdicts to `again`, `hard`, or
`good`; the model cannot return a schedule and one correct answer never becomes
`easy`. `ReviewScheduler` still calculates the next due time. The expected
answer, sources, and feedback are shown only after active recall and model
evaluation.

The answer, evaluation, trusted question snapshot, and resulting schedule form
an immutable `StudyAttempt`. Review calculation is separated from persistence so
the SQLite adapter writes the attempt and current progress in one transaction.
This prevents failed evaluation or persistence from advancing the schedule.
Attempts are append-only, identical retries are idempotent, and histories are
ordered chronologically for one exact question.

### Learning packages

Document ingestion deterministically classifies extracted text as German
(`de`), English (`en`), or unknown (`und`) without a network or model call. The
language is stored with `IndexedDocument`; existing SQLite libraries migrate to
`und` so legacy documents are not incorrectly relabeled. This source-language
metadata is deliberately separate from the package's learning-language choice.
Package creation stores that choice as
`same`, `de`, or `en` on both the durable preparation request and materialized
package. `same` remains the default for CLI calls and migrated databases. The
summarization service resolves `same` against the detected document language and
adds an explicit German or English instruction to every Map, Reduce, and repair
request. The language prompt reference is part of the generation identity, so
translated summaries cannot reuse source-language cache entries.

`LearningPackage` is the product-facing projection over one indexed document
and its active summary and question-bank identities. Its status records the
last successful expensive phase: `indexed`, `summarized`, or `ready`. Package
contents are references to existing versioned data rather than copies.

`LearningPackageService` coordinates the existing library, summarization, and
question-bank services through narrow preparation protocols. It saves after
every phase, so repeating `prepare` resumes after an interruption and a ready
package returns without loading models. `LearningPackageCatalog` provides a
read-only list for CLI and future UI clients. Package names are unique without
regard to case, while UUIDs remain internal identities.

`LearningPackageStudyService` is the product-facing study facade. It resolves a
case-insensitive package name, verifies that the package is ready, and delegates
due-question selection and written-answer recording to `StudySessionService`
with the exact stored document and question-bank identities. This keeps stable
technical provenance inside the application while normal learners select only a
library and package name. The technical positional `study` form remains
available for diagnostics and exact-version automation.

`LearningProgressService` resolves the same ready package and aggregates its
active question bank, current schedules, and immutable attempt histories. It
counts answered questions separately from attempts, because spaced repetition
can produce many attempts for one question. Verdict statistics use automatic
evaluations; legacy attempts without one are reported as `unclassified` rather
than inferred from self-ratings. Missing concepts are counted across evaluations
and ordered by frequency. New questions have no persisted schedule and are
therefore considered due at the report timestamp.

The progress report is a read-only projection. Creating it does not update review
schedules, persist derived analytics, or load embedding and generation models.
This keeps repeated status queries cheap and prevents observation from changing
the learning state.

For multi-batch summaries, Reduce must cite supported evidence from every Map
section. The application conservatively retains the complete validated Map
citation union because citations are currently global rather than attached to
individual summary claims.

### Grounded evaluation

`GroundedGenerationEvaluator` compares generated answers with versioned cases.
It reports exact source-page citation recall and precision plus deterministic
recall for curated concepts and accepted phrases. This is a reproducible
regression signal, not a substitute for human semantic or factual review.

## Command-line interface

`rag_learning_assistant.cli:main` remains the stable console entry point.

The CLI implementation is split as follows:

- `parser.py` composes the top-level parser and preserves its public imports;
- `parsing.py` contains shared constants, validators, and options;
- `parsers/packages.py`, `documents.py`, `retrieval.py`, `summaries.py`,
  `questions.py`, and `reviews.py` register responsibility-specific
  subcommands;
- `entrypoint.py` loads the optional environment, validates storage boundaries,
  dispatches commands, and translates application errors into CLI errors;
- `commands.py` wires concrete adapters and emits machine-readable JSON.

Current commands are:

```text
prepare
gui
package-list
progress
extract
index
list
remove
replace
search
ask
summarize
summary-list
summary-show
question-generate
question-list
question-show
review-due
review-record
study
```

A repository-root `.env` is loaded optionally. It may define `HF_TOKEN` for
authenticated Hugging Face downloads, but public-model commands continue when
the file is absent or environment loading fails.

## Local web interface

`rag-learn gui` starts a FastAPI application on `127.0.0.1`. The CLI resolves
the startup library directory. `LocalLibraryManager` treats its parent as a
bounded workspace, discovers only direct child directories containing
`metadata.sqlite3`, and delegates all web services to the selected library.
Each library has a `library.json` containing its internal UUID and independent
display name. New directories use the UUID rather than user input, so renaming
can never change a storage path. Existing libraries receive metadata in place
without moving their database. Create and open forms require the same loopback
origin as study submissions. Creating a library initializes an isolated SQLite
database but does not open it implicitly.

Management selection is page-local and does not change the library used by
package routes. Renaming atomically replaces only `library.json`. Deletion
re-resolves the UUID and workspace boundary immediately before removing the
directory, requires an exact display-name confirmation, and requires an
additional flag when SQLite rows or FAISS data show that the library is not
empty. Deleting the final library clears the optional service delegates and
hides package navigation. An empty workspace also remains empty on first
startup: the manager opens the configured startup directory only when it
already contains `metadata.sqlite3` and never initializes a library implicitly.

The package page exposes a multipart upload form as the input boundary for a
future preparation job. The server requires the loopback origin, a unique
display name, 1 to 50 questions, a `.pdf` filename, a PDF signature within the
first 1024 bytes, and at most 25 MiB. The same streaming pass calculates SHA-256
and rejects content already present in pending requests or indexed documents
before creating a visible request. `PackagePreparationService` atomically
stores accepted content in the selected library's `uploads` directory using a
UUID-derived filename; user input is retained only as display metadata. The
matching pending request is persisted in SQLite after the file move, and failed
registration removes both partial and completed upload files. This deliberately
separates upload ownership from later indexing and model-processing decisions.
The shared `package_names` registry reserves display names case-insensitively
across pending requests and materialized packages; the CLI checks it before
indexing to avoid orphaned document data. Preparation requests progress through
`pending`, `indexing`, `summarizing`, and `generating_questions`, or enter
`failed`. SQLite `BEGIN IMMEDIATE` claims one queued or expired request at a
time. A UUID lease token and expiry protect every phase transition from stale or
concurrent workers, while retry clears only the failure state and preserves the
uploaded PDF.
`WorkspacePreparationWorker` scans direct library directories and delegates one
request at a time to `PackagePreparationWorker`. The worker renews its lease on
a heartbeat thread during expensive local-model calls, transfers the shared
name reservation when indexing creates the materialized package, and advances
the persisted phase only after each package checkpoint exists. Completion
removes the request and uploaded PDF; failure records a bounded diagnostic and
keeps both the upload and last successful package checkpoint. The GUI server
starts this daemon worker for its lifetime. Model services are constructed only
after a request is claimed, so an idle GUI does not load embedding or generation
models. Active preparation prevents library deletion. While work is running, a
small browser-side poll replaces only the server-rendered package-list fragment;
the surrounding page and browser tab remain stable.
The worker passes the sanitized original filename separately to `LibraryService`.
Storage therefore remains UUID-based while extracted pages, persisted document
metadata, and citations retain the user-recognizable source name.
Successful upload uses Post/Redirect/Get to open that live package page instead
of rendering a stale validation snapshot. Failed cards translate known internal
exceptions into short user-facing explanations. Retry preserves checkpoints;
Remove invokes the normal package-removal lifecycle when indexing already
created a partial package, then deletes the request and owned upload.
Completed packages can be renamed or deleted from their detail page. Rename
updates the package row and shared case-insensitive name reservation atomically
while retaining the document and derived-material identities. Delete requires
an exact display-name confirmation and delegates to `LearningPackageService`
so document, retrieval, generated material, and learning records are removed by
the same lifecycle as the CLI.

The web application receives its service protocols through the factory instead
of importing CLI commands. Its start page is a library overview. Opening a
library redirects to its package page, while creation is isolated
on a management page. The package page renders preparation status and a
dedicated empty-library state. A package detail route uses the active persisted identities with
`DocumentSummaryCatalog` and `QuestionBankCatalog` to render the stored summary
and question count. Unknown names return a user-facing HTTP 404 page. Templates
and CSS are served from the installed package without external browser assets.
All pages extend `base.html`, which owns document metadata and a responsive,
server-rendered navigation bar. The shared template exposes library overview,
management, and the opened library's packages. JavaScript is limited to
progressive status feedback and input-state improvements; core navigation and
form actions remain server-rendered.

Ready package details link to a server-rendered study form. Its GET route asks
`LearningPackageStudyService` for the highest-priority due question without
revealing the expected answer. The POST route requires the request origin to
match the loopback application, verifies that the submitted question is still
the current due question, and then records the answer through the same atomic
evaluation and scheduling workflow as the CLI. Feedback reveals the expected
answer only after persistence succeeds. Trusted-host middleware rejects hosts
outside `127.0.0.1`, `localhost`, and the isolated test client.

Persisted review timestamps remain timezone-aware UTC values. The web result
converts the next review to the server machine's local timezone and labels it
as local time. UTC values remain available only in storage and technical output.

A small local progressive-enhancement script disables the submitted answer
button and exposes an ARIA live status while model evaluation is running. The
server-rendered form remains functional when browser scripting is unavailable.

The package detail also links to a read-only progress page. The web route depends
on a small `ProgressReporting` protocol and receives the existing
`LearningProgressService` from the server composition root, so CLI and GUI use
the same aggregation rules and SQLite repositories. User-facing timestamps are
converted from stored UTC values to the machine's local time before rendering.

## Persistence layout

Private documents and generated libraries belong under ignored local paths:

```text
local-data/
├── documents/
└── indexes/
    └── learning/
        ├── vectors.faiss
        └── metadata.sqlite3
```

`vectors.faiss` contains vector IDs and normalized embeddings.
`metadata.sqlite3` contains document and chunk metadata, embedding identity,
summary map cache entries, final summaries, question-generation batch entries,
question banks, learning packages, current question progress, and immutable
study attempts with optional automatic feedback. Its complete
[SQLite data model](database-overview.svg) documents columns, keys,
and logical relationships. The
nullable evaluation column migrates libraries created before automatic feedback.
Persistent formats validate identities when reopened so incompatible
models or generation configurations are not silently mixed.

## Environment setup

The package supports Python 3.11 through 3.13. Python 3.13 is the recommended
local version for the current Windows ML toolchain.

With Git Bash:

```bash
py -3.13 -m venv .venv && source .venv/Scripts/activate && python -m pip install --upgrade pip
```

Install all development and production extras:

```bash
python -m pip install -e ".[dev,embeddings,generation,storage]"
```

The extras are intentionally separated:

- `dev`: Pytest, coverage, and Ruff;
- `embeddings`: Sentence Transformers and its runtime dependencies;
- `generation`: Transformers, Accelerate, and PyTorch;
- `storage`: CPU FAISS. Embedding and generation models may still use CUDA.

### CUDA-enabled PyTorch on Windows

Installing ML packages from the regular package index can select a CPU-only
PyTorch build. Install the appropriate CUDA wheel before project extras:

```bash
python -m pip install --upgrade pip && python -m pip install torch==2.13.0 --index-url https://download.pytorch.org/whl/cu132 && python -m pip install -e ".[dev,embeddings,generation,storage]"
```

The current environment was verified with Python 3.13, PyTorch `2.13.0+cu132`,
CUDA 13.2, and an NVIDIA GeForce RTX 3060 Ti. Verify the active runtime:

```bash
python -c "import torch; print('Torch:', torch.__version__); print('CUDA:', torch.version.cuda); print('Available:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none')"
```

Use the [official PyTorch installation selector](https://pytorch.org/get-started/locally/)
when versions change. CUDA-specific wheels remain an environment concern and are
not pinned in `pyproject.toml`.

## Development workflow

The default workflow is red-green-refactor:

1. write one behavior-focused test;
2. verify the expected failure;
3. implement the smallest coherent behavior;
4. format, lint, and run the focused tests;
5. refactor only while tests remain green;
6. finish the milestone with the strict and isolated CI checks.

When multiple Git Bash commands are supplied, chain them with `&&` so later
checks stop after the first failure:

```bash
ruff format . && ruff check . && pytest -q
```

Before committing:

```bash
git status --short && git diff --check && git diff --stat
```

### Commit and milestone boundaries

A product milestone may consist of several small, coherent commits. Each commit
should represent one reviewable responsibility, such as application behavior,
CLI integration, or documentation. Tests that specify or verify that
responsibility belong in the same commit as its implementation. Documentation
may follow in its own commit when a user-facing workflow spans several code
changes.

Focused checks are sufficient while completing an individual commit, provided
the affected responsibility and its neighboring integration tests remain green.
The complete milestone and clean-environment checks are required after all
commits belonging to the milestone have been assembled and before the milestone
is tagged or released.

Project versions describe usable product releases, not individual commits. A
version is changed only when the assembled milestone forms a deliberate release.
After the full quality gates pass, that release may receive a matching Git tag.
Bug fixes or intermediate refactor commits do not automatically require a
version change or tag.

Run the complete local quality gate:

```bash
python scripts/check_milestone.py
```

This verifies Ruff formatting, linting, the complete suite, at least 90% coverage,
resource cleanup warnings, unraisable exceptions, and Git whitespace.

Reproduce the GitHub CI dependency boundary in a disposable virtual environment:

```bash
python scripts/check_ci_environment.py
```

The temporary environment is deleted automatically. It installs only
`dev,storage`, verifies that tooling imports without Torch, and runs the shared
milestone gate. This catches missing optional-dependency boundaries before code
is pushed.

## Testing strategy

Tests mirror production responsibilities:

```text
tests/
├── application/
├── chunking/
├── evaluation/
├── generation/
├── ingestion/
├── interfaces/cli/
├── learning/
├── library/
├── retrieval/
├── scripts/
└── test_cli.py
```

Fast tests use fakes at model, filesystem, and application boundaries. The
default suite must remain offline-capable and must not download or load real ML
models. SQLite and FAISS tests use temporary libraries and reopen them to verify
durability.

`--import-mode=importlib` permits repeated test filenames in responsibility-based
directories. CI runs Python 3.11, 3.12, and 3.13 within the package's declared
support range.

## Manual smoke tests and benchmarks

Hardware-, network-, and model-dependent checks remain separate from CI.

### Generation adapter

```bash
python scripts/smoke_test_generation.py
```

This loads the real pinned Qwen model, verifies CUDA, produces strict structured
output for a controlled context, checks citation `(1,)`, and reports peak GPU
memory.

### Full RAG answer

```bash
rag-learn ask local-data/indexes/learning "What are Python functions?" --limit 3
```

This exercises query embedding, FAISS retrieval, prompt construction, local
generation, response parsing, and trusted citation mapping.

### Document summarization

```bash
python scripts/smoke_test_summarization.py INDEX_DIRECTORY DOCUMENT_UUID --max-map-new-tokens 192 --max-reduce-new-tokens 384 --max-batch-chars 8000
```

The script prints map/reduce progress, JSON output, elapsed time, GPU name, and
peak memory. A reproducible redistributable fixture and measurements are
described in the [summarization benchmark](summarization-benchmark.md).

For detailed call timings and cache counters:

```bash
python scripts/benchmark_summarization.py INDEX_DIRECTORY DOCUMENT_UUID --max-map-new-tokens 192 --max-reduce-new-tokens 384 --max-batch-chars 8000 --ignore-cache
```

### Grounded generation evaluation

```bash
python scripts/evaluate_grounded_generation.py local-data/indexes/summarization-benchmark --output local-data/evaluation/grounded-generation-baseline.json
```

The report includes per-case citation and concept results plus aggregate recall,
precision, and pass rate.

## Extension points

### Embedding provider

Implement `Embedder` with separate `embed_documents` and `embed_query` methods.
Provider-specific prefixes, batching, clients, and response conversion stay in
the adapter. `RetrievalService` must remain provider-agnostic.

### Vector store

Implement `VectorStore` and preserve enough model metadata to reject incompatible
indices. Persistent stores must keep vector IDs and source metadata consistent
across add, remove, replace, reopen, and failed writes.

### Text-generation provider

Implement `TextGenerator` for grounded answers and summaries. Question generation
also requires the narrow `QuestionGenerator` behavior. Provider adapters must
return validated `GenerationResult` or `QuestionGenerationResult` objects and
report prompt references actually used. Application services remain responsible
for mapping citation numbers back to trusted source objects.

### Learning algorithm

Keep scheduling policy inside `ReviewScheduler`. A future replacement may use a
different algorithm, but it must preserve timezone-aware due dates, immutable
state transitions, the minimum ease invariant, and deterministic tests. If new
state fields are required, add an explicit SQLite migration rather than silently
changing stored meaning.

## Documentation and decisions

ADRs explain why durable choices were made. `development.md` describes the
current integrated system. Code comments explain non-obvious local constraints,
and tests remain the executable behavior specification.

Update the relevant [PlantUML class diagram](diagrams/README.md) whenever classes,
protocols, inheritance, or major runtime relationships change.

## Near-term roadmap

1. Add a local web interface for package discovery and study workflows.
2. Add manual correction for incorrectly evaluated study answers.
3. Add progress analytics over the append-only attempt history.
4. Generate detailed section-level learning material.
5. Add optional Ollama and remote API generation providers.
