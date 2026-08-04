import json
from pathlib import Path
from uuid import UUID

import pytest

from rag_learning_assistant import cli
from rag_learning_assistant.application import DuplicateDocumentError
from rag_learning_assistant.chunking import Chunk
from rag_learning_assistant.ingestion import Document, Page
from rag_learning_assistant.interfaces.cli import commands, entrypoint
from rag_learning_assistant.interfaces.cli.parser import (
    build_parser,
    validate_index_directory,
)
from rag_learning_assistant.library import IndexedDocument
from rag_learning_assistant.retrieval import SearchResult


class FakeDocumentSearchService:
    def __init__(self, results: list[SearchResult]) -> None:
        self.results = results
        self.indexed_documents: list[Document] = []
        self.search_calls: list[tuple[str, int]] = []

    def index_document(self, document: Document) -> list[Chunk]:
        self.indexed_documents.append(document)
        return []

    def search(self, query: str, limit: int) -> list[SearchResult]:
        self.search_calls.append((query, limit))
        return self.results


class FakeLibraryService:
    def __init__(self, document: IndexedDocument) -> None:
        self.document = document
        self.paths: list[Path] = []

    def add_document(self, path: Path) -> IndexedDocument:
        self.paths.append(path)
        return self.document


class FakeLibraryCatalog:
    def __init__(self, documents: list[IndexedDocument]) -> None:
        self.documents = documents
        self.list_calls = 0

    def list_documents(self) -> list[IndexedDocument]:
        self.list_calls += 1
        return list(self.documents)


def test_cli_outputs_machine_readable_json(monkeypatch, tmp_path: Path, capsys) -> None:
    pdf = tmp_path / "course.pdf"
    pdf.touch()
    document = Document("course.pdf", (Page(1, "Lesson", "course.pdf"),))
    monkeypatch.setattr(entrypoint.PdfExtractor, "extract", lambda self, path: document)

    assert cli.main(["extract", str(pdf)]) == 0
    assert json.loads(capsys.readouterr().out) == {
        "source": "course.pdf",
        "pages": [
            {
                "number": 1,
                "source": "course.pdf",
                "text": "Lesson",
            }
        ],
        "chunks": [
            {
                "index": 0,
                "text": "Lesson",
                "source": "course.pdf",
                "page_number": 1,
            }
        ],
    }


def test_cli_accepts_chunking_options(monkeypatch, tmp_path: Path, capsys) -> None:
    pdf = tmp_path / "course.pdf"
    pdf.touch()
    document = Document(
        "course.pdf",
        (Page(1, "one two three", "course.pdf"),),
    )
    monkeypatch.setattr(entrypoint.PdfExtractor, "extract", lambda self, path: document)

    result = cli.main(
        [
            "extract",
            str(pdf),
            "--max-chars",
            "7",
            "--overlap-chars",
            "0",
        ]
    )

    payload = json.loads(capsys.readouterr().out)

    assert result == 0
    assert [chunk["text"] for chunk in payload["chunks"]] == [
        "one two",
        "three",
    ]


def test_cli_rejects_invalid_chunking_options(
    monkeypatch,
    tmp_path: Path,
    capsys,
) -> None:
    pdf = tmp_path / "course.pdf"
    pdf.touch()
    document = Document(
        "course.pdf",
        (Page(1, "Lesson", "course.pdf"),),
    )
    monkeypatch.setattr(entrypoint.PdfExtractor, "extract", lambda self, path: document)

    with pytest.raises(SystemExit) as exc_info:
        cli.main(
            [
                "extract",
                str(pdf),
                "--max-chars",
                "100",
                "--overlap-chars",
                "100",
            ]
        )

    assert exc_info.value.code == 2
    assert "overlap_chars must be smaller than max_chars" in capsys.readouterr().err


def test_cli_reports_duplicate_document(
    monkeypatch,
    tmp_path: Path,
    capsys,
) -> None:
    pdf = tmp_path / "duplicate.pdf"
    pdf.touch()
    index_directory = tmp_path / "learning-library"

    class DuplicateLibraryService:
        def add_document(self, path: Path) -> IndexedDocument:
            raise DuplicateDocumentError("Document content is already indexed as original.pdf")

    monkeypatch.setattr(
        commands,
        "build_library_service",
        lambda chunker, index_dir: DuplicateLibraryService(),
    )

    result = cli.main(
        [
            "index",
            str(pdf),
            "--index-dir",
            str(index_directory),
        ]
    )

    payload = json.loads(capsys.readouterr().out)

    assert result == 0
    assert payload["results"] == [
        {
            "path": str(pdf),
            "status": "skipped",
            "document": None,
            "message": "Document content is already indexed as original.pdf",
        }
    ]
    assert payload["summary"] == {
        "added": 0,
        "skipped": 1,
        "failed": 0,
    }


def test_cli_search_outputs_ranked_results(
    monkeypatch,
    tmp_path: Path,
    capsys,
) -> None:
    index_directory = tmp_path / "course-index"
    index_directory.mkdir()
    (index_directory / "vectors.faiss").touch()
    (index_directory / "metadata.sqlite3").touch()

    chunk = Chunk(
        text="Python functions",
        source="course.pdf",
        page_number=4,
        index=0,
    )
    search_service = FakeDocumentSearchService([SearchResult(chunk=chunk, score=0.91)])

    monkeypatch.setattr(
        commands,
        "build_persistent_retrieval",
        lambda index_dir: search_service,
    )

    result = cli.main(
        [
            "search",
            str(index_directory),
            "How do Python functions work?",
            "--limit",
            "3",
        ]
    )

    assert result == 0
    assert search_service.search_calls == [("How do Python functions work?", 3)]
    assert json.loads(capsys.readouterr().out) == {
        "query": "How do Python functions work?",
        "results": [
            {
                "score": 0.91,
                "text": "Python functions",
                "source": "course.pdf",
                "page_number": 4,
                "index": 0,
            }
        ],
    }


def test_cli_index_registers_library_document(
    monkeypatch,
    tmp_path: Path,
    capsys,
) -> None:
    pdf = tmp_path / "course.pdf"
    pdf.touch()
    index_directory = tmp_path / "course-index"
    indexed_document = IndexedDocument(
        id=UUID("12345678-1234-5678-1234-567812345678"),
        source="course.pdf",
        content_sha256="a" * 64,
        page_count=2,
        chunk_count=3,
    )
    library_service = FakeLibraryService(indexed_document)

    def fail_if_entrypoint_extracts_pdf(self, path):
        raise AssertionError("LibraryService must coordinate PDF extraction")

    monkeypatch.setattr(
        entrypoint.PdfExtractor,
        "extract",
        fail_if_entrypoint_extracts_pdf,
    )
    monkeypatch.setattr(
        commands,
        "build_library_service",
        lambda chunker, index_dir: library_service,
        raising=False,
    )

    result = cli.main(
        [
            "index",
            str(pdf),
            "--index-dir",
            str(index_directory),
        ]
    )

    assert result == 0
    assert library_service.paths == [pdf]
    assert json.loads(capsys.readouterr().out) == {
        "index_directory": str(index_directory),
        "results": [
            {
                "path": str(pdf),
                "status": "added",
                "document": {
                    "id": "12345678-1234-5678-1234-567812345678",
                    "source": "course.pdf",
                    "content_sha256": "a" * 64,
                    "page_count": 2,
                    "chunk_count": 3,
                },
                "message": None,
            }
        ],
        "summary": {
            "added": 1,
            "skipped": 0,
            "failed": 0,
        },
    }


def test_cli_indexes_multiple_documents_in_one_batch(
    monkeypatch,
    tmp_path: Path,
    capsys,
) -> None:
    first_pdf = tmp_path / "first.pdf"
    second_pdf = tmp_path / "second.pdf"
    first_pdf.touch()
    second_pdf.touch()
    index_directory = tmp_path / "learning-library"
    document = IndexedDocument(
        id=UUID("12345678-1234-5678-1234-567812345678"),
        source="book.pdf",
        content_sha256="a" * 64,
        page_count=2,
        chunk_count=3,
    )
    library_service = FakeLibraryService(document)

    monkeypatch.setattr(
        commands,
        "build_library_service",
        lambda chunker, index_dir: library_service,
    )

    result = cli.main(
        [
            "index",
            str(first_pdf),
            str(second_pdf),
            "--index-dir",
            str(index_directory),
        ]
    )

    payload = json.loads(capsys.readouterr().out)

    assert result == 0
    assert library_service.paths == [first_pdf, second_pdf]
    assert [item["status"] for item in payload["results"]] == [
        "added",
        "added",
    ]
    assert payload["summary"] == {
        "added": 2,
        "skipped": 0,
        "failed": 0,
    }


def test_cli_batch_returns_one_when_a_document_fails(
    monkeypatch,
    tmp_path: Path,
    capsys,
) -> None:
    pdf = tmp_path / "broken.pdf"
    pdf.touch()
    index_directory = tmp_path / "learning-library"

    class FailingLibraryService:
        def add_document(self, path: Path) -> IndexedDocument:
            raise ValueError("Could not open PDF")

    monkeypatch.setattr(
        commands,
        "build_library_service",
        lambda chunker, index_dir: FailingLibraryService(),
    )

    result = cli.main(
        [
            "index",
            str(pdf),
            "--index-dir",
            str(index_directory),
        ]
    )

    payload = json.loads(capsys.readouterr().out)

    assert result == 1
    assert payload["results"] == [
        {
            "path": str(pdf),
            "status": "failed",
            "document": None,
            "message": "Could not open PDF",
        }
    ]
    assert payload["summary"]["failed"] == 1


def test_cli_index_accepts_existing_library(
    monkeypatch,
    tmp_path: Path,
    capsys,
) -> None:
    pdf = tmp_path / "second-book.pdf"
    pdf.touch()
    index_directory = tmp_path / "learning-library"
    index_directory.mkdir()
    (index_directory / "vectors.faiss").touch()
    (index_directory / "metadata.sqlite3").touch()
    indexed_document = IndexedDocument(
        id=UUID("12345678-1234-5678-1234-567812345678"),
        source="second-book.pdf",
        content_sha256="b" * 64,
        page_count=10,
        chunk_count=25,
    )
    library_service = FakeLibraryService(indexed_document)

    monkeypatch.setattr(
        commands,
        "build_library_service",
        lambda chunker, index_dir: library_service,
    )

    result = cli.main(
        [
            "index",
            str(pdf),
            "--index-dir",
            str(index_directory),
        ]
    )

    assert result == 0
    assert library_service.paths == [pdf]
    payload = json.loads(capsys.readouterr().out)
    assert payload["results"][0]["document"]["source"] == "second-book.pdf"


def test_cli_lists_library_documents(
    monkeypatch,
    tmp_path: Path,
    capsys,
) -> None:
    index_directory = tmp_path / "learning-library"
    index_directory.mkdir()
    (index_directory / "metadata.sqlite3").touch()
    document = IndexedDocument(
        id=UUID("12345678-1234-5678-1234-567812345678"),
        source="python-book.pdf",
        content_sha256="a" * 64,
        page_count=120,
        chunk_count=758,
    )
    catalog = FakeLibraryCatalog([document])

    monkeypatch.setattr(
        commands,
        "build_library_catalog",
        lambda index_dir: catalog,
        raising=False,
    )

    result = cli.main(
        [
            "list",
            str(index_directory),
        ]
    )

    assert result == 0
    assert catalog.list_calls == 1
    assert json.loads(capsys.readouterr().out) == {
        "index_directory": str(index_directory),
        "documents": [
            {
                "id": "12345678-1234-5678-1234-567812345678",
                "source": "python-book.pdf",
                "content_sha256": "a" * 64,
                "page_count": 120,
                "chunk_count": 758,
            }
        ],
    }


def test_parser_accepts_search_command() -> None:
    args = build_parser().parse_args(
        [
            "search",
            ".rag-index/book",
            "What are Python functions?",
            "--limit",
            "3",
        ]
    )

    assert args.command == "search"
    assert args.index_dir == Path(".rag-index/book")
    assert args.query == "What are Python functions?"
    assert args.limit == 3
    assert not hasattr(args, "pdf")


@pytest.mark.parametrize("limit", ["0", "-1"])
def test_search_limit_must_be_positive(
    limit: str,
    tmp_path: Path,
    capsys,
) -> None:
    index_directory = tmp_path / "course-index"

    with pytest.raises(SystemExit) as exc_info:
        cli.main(
            [
                "search",
                str(index_directory),
                "What is Python?",
                "--limit",
                limit,
            ]
        )

    assert exc_info.value.code == 2
    assert "must be positive" in capsys.readouterr().err


@pytest.mark.parametrize("query", ["", "   "])
def test_search_query_must_not_be_blank(
    query: str,
    tmp_path: Path,
    capsys,
) -> None:
    index_directory = tmp_path / "course-index"

    with pytest.raises(SystemExit) as exc_info:
        cli.main(
            [
                "search",
                str(index_directory),
                query,
            ]
        )

    assert exc_info.value.code == 2
    assert "must not be blank" in capsys.readouterr().err


def test_parser_accepts_multiple_index_documents() -> None:
    args = build_parser().parse_args(
        [
            "index",
            "first.pdf",
            "second.pdf",
            "--index-dir",
            "local-data/indexes/learning",
        ]
    )

    assert args.command == "index"
    assert args.pdfs == [
        Path("first.pdf"),
        Path("second.pdf"),
    ]
    assert args.index_dir == Path("local-data/indexes/learning")
    assert not hasattr(args, "pdf")


def test_index_rejects_incomplete_index_directory(
    monkeypatch,
    tmp_path: Path,
    capsys,
) -> None:
    pdf = tmp_path / "course.pdf"
    pdf.touch()
    index_directory = tmp_path / "course-index"
    index_directory.mkdir()
    (index_directory / "vectors.faiss").touch()

    def fail_if_pdf_is_opened(self, path):
        raise AssertionError("PDF must not be opened for an invalid index directory")

    monkeypatch.setattr(
        entrypoint.PdfExtractor,
        "extract",
        fail_if_pdf_is_opened,
    )

    with pytest.raises(SystemExit) as exc_info:
        cli.main(
            [
                "index",
                str(pdf),
                "--index-dir",
                str(index_directory),
            ]
        )

    assert exc_info.value.code == 2
    assert "index directory is incomplete" in capsys.readouterr().err


def test_index_accepts_metadata_only_directory_after_failed_import(
    tmp_path: Path,
) -> None:
    index_directory = tmp_path / "course-index"
    index_directory.mkdir()
    (index_directory / "metadata.sqlite3").touch()

    validate_index_directory(index_directory)


def test_search_rejects_missing_index_directory(
    monkeypatch,
    tmp_path: Path,
    capsys,
) -> None:
    missing_index = tmp_path / "missing-index"

    def fail_if_retrieval_is_built(index_directory):
        raise AssertionError("Retrieval must not be built for a missing index")

    monkeypatch.setattr(
        commands,
        "build_persistent_retrieval",
        fail_if_retrieval_is_built,
    )

    with pytest.raises(SystemExit) as exc_info:
        cli.main(
            [
                "search",
                str(missing_index),
                "What is Python?",
            ]
        )

    assert exc_info.value.code == 2
    assert "index directory is incomplete" in capsys.readouterr().err


def test_parser_accepts_list_command() -> None:
    args = build_parser().parse_args(
        [
            "list",
            "local-data/indexes/learning",
        ]
    )

    assert args.command == "list"
    assert args.index_dir == Path("local-data/indexes/learning")
