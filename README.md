# RAG Learning Assistant

An open-source learning assistant that turns your PDF documents into source-grounded
learning material, questions, and interactive tutoring sessions.

The project is at an early MVP stage. The first milestone extracts text page by page and
preserves the metadata needed for citations. Retrieval, embeddings, question generation,
and learning-progress tracking will build on this foundation.

## Current features

- extract text from text-based PDF files
- retain document name and one-based page numbers
- output structured JSON through a small command-line interface
- testable core with no model or cloud-service dependency

Scanned documents do not yet support OCR.

## Quick start

Python 3.11 or newer is required.

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
pytest
```

Extract a PDF:

```powershell
rag-learn path\to\book.pdf
```

The command emits JSON containing the text, source file, and page number for every page.

## Planned architecture

```text
PDF -> page extraction -> semantic chunks -> embeddings -> vector search
                                                        -> grounded LLM tutor
                                                        -> learning materials
                                                        -> questions and progress
```

Planned next milestones:

1. structure-aware chunking with page citations
2. local embeddings and vector retrieval
3. answers with quoted source references
4. summaries and reusable question banks
5. learner feedback and spaced repetition
6. optional Ollama and API model providers

## Development

Run the checks locally:

```powershell
pytest --cov=rag_learning_assistant
ruff check .
ruff format --check .
```

Contributions are welcome. Please read [CONTRIBUTING.md](CONTRIBUTING.md).

## License

This project is available under the [MIT License](LICENSE).

