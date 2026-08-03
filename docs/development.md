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
├── application/
│   └── document_search.py
├── ingestion/
│   ├── models.py
│   └── pdf.py
├── chunking/
│   ├── models.py
│   └── service.py
└── retrieval/
    ├── embeddings.py
    ├── faiss_store.py
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

`InMemoryVectorStore` is the reference implementation.
It uses cosine similarity and validates vector dimensions and non-zero magnitudes.
`RetrievalService` coordinates batch indexing and query search.

`FaissVectorStore` is the persistent implementation.
It stores normalized vectors and numeric IDs in an exact FAISS inner-product index and stores chunks plus embedding-model metadata in SQLite.
For normalized vectors, inner product is equivalent to cosine similarity.
Batch writes load and persist the FAISS index once per document instead of once per chunk.

`SentenceTransformerEmbedder` provides local embeddings using the pinned default model:

```text
Model:    intfloat/multilingual-e5-small
Revision: 614241f622f53c4eeff9890bdc4f31cfecc418b3
Dimension: 384
```

The adapter adds the E5 `passage: ` and `query: ` prefixes, requests normalized vectors, and loads the model only on first use.
The model is cached by Hugging Face outside the repository.

### Application flow

`DocumentSearchService` coordinates chunking and indexing without implementing those responsibilities itself.
The CLI exposes `extract`, `index`, and `search` as separate commands.
`index` processes a PDF into a new persistent index, while `search` opens that index without reopening or re-embedding the PDF.
Index paths are validated before a PDF or model is loaded.

## Public interfaces

Each feature package exports its supported public API from `__init__.py`.
Internal code may import concrete modules directly to keep dependencies explicit, while callers should prefer package-level imports:

```python
from rag_learning_assistant.chunking import Chunk, TextChunker
from rag_learning_assistant.ingestion import Document, Page, PdfExtractor
from rag_learning_assistant.retrieval import RetrievalService
```

## Environment setup

Python 3.11 or newer is supported by the package.
Python 3.13 is the recommended local version for the current ML toolchain.
On Windows with Git Bash:

```bash
py -3.13 -m venv .venv && source .venv/Scripts/activate
```

Install the core development dependencies:

```bash
python -m pip install -e ".[dev]"
```

Install local Hugging Face embedding and persistent-storage support as well:

```bash
python -m pip install -e ".[dev,embeddings,storage]"
```

The embedding extra includes Sentence Transformers and its ML runtime dependencies.
The storage extra includes the CPU FAISS runtime; embedding generation can still use CUDA independently.
The core PDF and chunking functionality deliberately remains usable without them.

## Persistent indexing

Keep private documents and generated indices outside version control:

```text
local-data/
├── documents/
└── indexes/
```

Create and query an index:

```bash
rag-learn index local-data/documents/book.pdf --index-dir local-data/indexes/book
rag-learn search local-data/indexes/book "What are Python functions?" --limit 3
```

An index directory contains:

```text
book/
├── vectors.faiss
└── metadata.sqlite3
```

SQLite maps persistent FAISS IDs to chunk text, source, page number, and document-wide chunk index.
It also records model name, pinned revision, and the dimension established by the first embedding.
Opening an index with a different model identity is rejected.

Indexing currently accepts only a new or empty target directory.
This intentionally prevents duplicate chunks until document identity and update semantics are implemented.

### CUDA-enabled PyTorch on Windows

Installing Sentence Transformers from the regular Python package index can resolve to a CPU-only PyTorch build.
Changing the Python version alone does not enable GPU execution.

For a supported NVIDIA GPU, install the appropriate CUDA-enabled PyTorch wheel before the project extras:

```bash
python -m pip install --upgrade pip && python -m pip install torch==2.13.0 --index-url https://download.pytorch.org/whl/cu132
python -m pip install -e ".[dev,embeddings,storage]"
```

This order lets the Sentence Transformers dependency reuse the already installed CUDA-enabled PyTorch version.
The current setup was verified on Windows with Python 3.13, PyTorch `2.13.0+cu132`, CUDA 13.2, and an NVIDIA GeForce RTX 3060 Ti.

Check the runtime after installation:

```bash
python -c "import torch; print('Torch:', torch.__version__); print('CUDA:', torch.version.cuda); print('Available:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none')"
```

Expected key values for the verified setup are `CUDA: 13.2` and `Available: True`.
Use the [official PyTorch installation selector](https://pytorch.org/get-started/locally/) when changing versions because CUDA wheel availability is maintained separately from `pyproject.toml`.

The project metadata intentionally does not pin a CUDA-specific PyTorch build.
CPU, CUDA, and other accelerator variants require different package indexes, so the runtime choice remains an environment setup concern.

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
The complete CLI flow was then smoke-tested with a real text-based PDF and returned three semantically relevant, page-cited results.

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

    def add_many(
        self,
        entries: Sequence[tuple[Chunk, Embedding]],
    ) -> None: ...

    def search(self, query: Embedding, limit: int) -> list[SearchResult]: ...
```

Persistent implementations must store enough model metadata to detect incompatible indices.
Changing the embedding model or revision requires rebuilding existing vectors.

## Documentation and decisions

Long-lived decisions are recorded in [Architecture Decision Records](adr/README.md).
ADRs explain context and consequences; code comments should explain only non-obvious local decisions.
Tests remain the executable specification of behavior.

## Near-term roadmap

1. Define document identity and index-update semantics.
2. Build a small German retrieval evaluation set.
3. Add grounded answer generation with citations.
4. Add reusable learning-material and question-generation workflows.
