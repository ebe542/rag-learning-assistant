# RAG Learning Assistant

An open-source learning assistant that turns your PDF documents into source-grounded
learning material, questions, and interactive tutoring sessions.

The project is at an early MVP stage. It extracts text page by page, preserves citation
metadata, and creates configurable, paragraph-aware chunks. Retrieval, embeddings,
question generation, and learning-progress tracking will build on this foundation.

## Current features

- extract text from text-based PDF files
- retain document name and one-based page numbers
- output structured JSON through a small command-line interface
- testable core with no model or cloud-service dependency
- split pages into size-limited, overlapping chunks while preserving paragraph boundaries

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
rag-learn path\to\book.pdf --max-chars 1000 --overlap-chars 150
```

The command emits JSON containing the extracted pages and searchable chunks. Every chunk
retains its source file, page number, and document-wide index. Chunk size and overlap can
be configured through the command-line options shown above.

## Planned architecture

```text
PDF -> page extraction -> semantic chunks -> embeddings -> vector search
                                                        -> grounded LLM tutor
                                                        -> learning materials
                                                        -> questions and progress
```

Planned next milestones:

1. local embeddings and vector retrieval
2. answers with quoted source references
3. summaries and reusable question banks
4. learner feedback and spaced repetition
5. optional Ollama and API model providers

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
