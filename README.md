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
- search an in-memory vector store using cosine similarity
- create local multilingual E5 embeddings through an optional Sentence Transformers adapter

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

Install the optional local embedding support:

```powershell
python -m pip install -e ".[dev,embeddings]"
```

### NVIDIA GPU support on Windows

The regular Python package index may install a CPU-only PyTorch build.
For an NVIDIA GPU, install the CUDA-enabled PyTorch build before installing the embedding dependencies.
The following setup was verified with Python 3.13, PyTorch 2.13.0, CUDA 13.2, and an NVIDIA GeForce RTX 3060 Ti:

```bash
py -3.13 -m venv .venv && source .venv/Scripts/activate
python -m pip install --upgrade pip && python -m pip install torch==2.13.0 --index-url https://download.pytorch.org/whl/cu132
python -m pip install -e ".[dev,embeddings]"
```

Verify that PyTorch can use the GPU:

```bash
python -c "import torch; print('Torch:', torch.__version__); print('CUDA:', torch.version.cuda); print('Available:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none')"
```

The CUDA wheel must match the installed NVIDIA driver and a version offered by PyTorch.
Refer to the [official PyTorch installation instructions](https://pytorch.org/get-started/locally/) when changing PyTorch or CUDA versions.

The default embedding model is `intfloat/multilingual-e5-small`, pinned to a specific Hugging Face revision for reproducibility.
The first real embedding request downloads the model into the local Hugging Face cache.

Extract pages and chunks from a PDF:

```powershell
rag-learn extract path\to\book.pdf --max-chars 1000 --overlap-chars 150
```

The command emits JSON containing the extracted pages and searchable chunks.
Every chunk retains its source file, page number, and document-wide index.
Chunk size and overlap can be configured through the command-line options shown above.

Search the document semantically with the local embedding model:

```powershell
rag-learn search path\to\book.pdf "What are Python functions?" --limit 3
```

The current in-memory workflow extracts, chunks, and embeds the PDF for every search invocation.
Persistent indexing is planned as the next storage milestone.

## Planned architecture

```text
PDF -> page extraction -> semantic chunks -> embeddings -> vector search
                                                        -> grounded LLM tutor
                                                        -> learning materials
                                                        -> questions and progress
```

Planned next milestones:

1. persist embeddings and document metadata locally
2. answer questions with quoted source references
3. generate summaries and reusable question banks
4. track learner feedback and spaced repetition
5. add optional Ollama and API model providers

## Development

Run the checks locally:

```powershell
pytest --cov=rag_learning_assistant
ruff check .
ruff format --check .
```

Contributions are welcome.
Please read [CONTRIBUTING.md](CONTRIBUTING.md).
For architecture and extension guidance, see the [development guide](docs/development.md) and [architecture decisions](docs/adr/README.md).

## License

This project is available under the [MIT License](LICENSE).
