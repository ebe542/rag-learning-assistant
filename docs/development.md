# Development Guide

This guide describes the architecture, local workflow, and extension points of the RAG Learning Assistant.
For user-facing setup and usage, see the project [README](../README.md).

## Architecture

The application is built as a sequence of independent stages:

```text
PDF -> ingestion -> chunking -> embeddings -> vector retrieval -> grounded generation
```

The current source tree reflects implemented responsibilities rather than anticipated features:

```text
src/rag_learning_assistant/
├── cli.py
├── ingestion/
│   ├── models.py
│   └── pdf.py
├── chunking/
│   ├── models.py
│   └── service.py
└── retrieval/
    ├── embeddings.py
    ├── models.py
    ├── sentence_transformer.py
    ├── service.py
    └── store.py
```

### Ingestion

`PdfExtractor` reads text-based PDFs with PyMuPDF.
It produces immutable `Document` and `Page` models while retaining one-based page numbers and the source filename.
Scanned PDFs do not yet support OCR.

### Chunking

`TextChunker` turns pages into immutable `Chunk` objects.
Chunks never cross page boundaries, and their indices increase across the entire document.

The chunker:

- prefers complete paragraph boundaries;
- splits long paragraphs at word boundaries;
- hard-splits words longer than the configured maximum;
- applies overlap only when a long paragraph has to be split;
- rejects invalid size and overlap settings.

Chunk size is currently measured in characters to keep the core independent of a specific model and tokenizer.

### Retrieval

`Embedder` separates document and query embeddings because asymmetric retrieval models may require different input representations.
`VectorStore` defines storage and search without coupling the application service to a database implementation.

`InMemoryVectorStore` is the first implementation.
It uses cosine similarity and validates vector dimensions and non-zero magnitudes.
`RetrievalService` coordinates batch indexing and query search.

`SentenceTransformerEmbedder` provides local embeddings using the pinned default model:

```text
Model:    intfloat/multilingual-e5-small
Revision: 614241f622f53c4eeff9890bdc4f31cfecc418b3
Dimension: 384
```

The adapter adds the E5 `passage: ` and `query: ` prefixes, requests normalized vectors, and loads the model only on first use.
The model is cached by Hugging Face outside the repository.

## Public interfaces

Each feature package exports its supported public API from `__init__.py`.
Internal code may import concrete modules directly to keep dependencies explicit, while callers should prefer package-level imports:

```python
from rag_learning_assistant.chunking import Chunk, TextChunker
from rag_learning_assistant.ingestion import Document, Page, PdfExtractor
from rag_learning_assistant.retrieval import RetrievalService
```

## Environment setup

Python 3.11 or newer is required.
On Windows with Git Bash:

```bash
python -m venv .venv && source .venv/Scripts/activate
```

Install the core development dependencies:

```bash
python -m pip install -e ".[dev]"
```

Install local Hugging Face embedding support as well:

```bash
python -m pip install -e ".[dev,embeddings]"
```

The embedding extra includes Sentence Transformers and its ML runtime dependencies.
The core PDF and chunking functionality deliberately remains usable without them.

## Quality checks

Use the following command during local development:

```bash
ruff format . && ruff check . && pytest -q
```

Before committing, also inspect whitespace and scope:

```bash
git status --short && git diff --check && git diff --stat
```

CI runs tests with coverage and verifies Ruff formatting without modifying files.
Pytest uses `--import-mode=importlib`, allowing identically named test modules in different responsibility-based directories.

## Testing strategy

Behavior is generally developed using a red-green-refactor cycle.
Unit tests use small fakes at integration boundaries so the default test suite never downloads models or depends on external services.

The real `multilingual-e5-small` adapter was additionally smoke-tested in German.
This manual check confirmed 384-dimensional vectors and ranked a Python passage above an unrelated database passage for a question about Python functions.

Tests mirror the production structure:

```text
tests/
├── test_cli.py
├── ingestion/
├── chunking/
└── retrieval/
```

## Adding an embedding provider

A provider implements the `Embedder` protocol:

```python
class Embedder(Protocol):
    def embed_documents(self, texts: Sequence[str]) -> list[Embedding]: ...

    def embed_query(self, text: str) -> Embedding: ...
```

Provider-specific prefixes, batching, network clients, and response conversion belong inside the adapter.
`RetrievalService` must remain unaware of those details.

Add contract-focused unit tests with a fake backend.
Network-dependent smoke tests should not be part of the fast default suite.

## Adding a vector store

A store implements the `VectorStore` protocol:

```python
class VectorStore(Protocol):
    def add(self, chunk: Chunk, embedding: Embedding) -> None: ...

    def search(self, query: Embedding, limit: int) -> list[SearchResult]: ...
```

Persistent implementations must store enough model metadata to detect incompatible indices.
Changing the embedding model or revision requires rebuilding existing vectors.

## Documentation and decisions

Long-lived decisions are recorded in [Architecture Decision Records](adr/README.md).
ADRs explain context and consequences; code comments should explain only non-obvious local decisions.
Tests remain the executable specification of behavior.

## Near-term roadmap

1. Connect PDF ingestion, chunking, E5 embeddings, and retrieval in one application flow.
2. Add semantic search to the CLI.
3. Persist vectors, chunks, and embedding-model metadata locally.
4. Build a small German retrieval evaluation set.
5. Add grounded answer generation with citations.
