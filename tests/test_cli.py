import json
from importlib.metadata import version
from pathlib import Path
from uuid import UUID

import pytest

from rag_learning_assistant import cli
from rag_learning_assistant.application import (
    DocumentNotFoundError,
    DuplicateDocumentError,
)
from rag_learning_assistant.chunking import Chunk
from rag_learning_assistant.ingestion import Document, Page
from rag_learning_assistant.interfaces.cli import commands
from rag_learning_assistant.interfaces.cli.parser import (
    build_parser,
    validate_index_directory,
)
from rag_learning_assistant.library import IndexedDocument
from rag_learning_assistant.retrieval import SearchResult


class FakePdfExtractor:
    def __init__(self, document: Document) -> None:
        self.document = document
        self.paths: list[Path] = []

    def extract(
        self,
        path: str | Path,
    ) -> Document:
        self.paths.append(Path(path))
        return self.document


def test_cli_reports_installed_package_version(capsys) -> None:
    with pytest.raises(SystemExit) as exc_info:
        cli.main(["--version"])

    assert exc_info.value.code == 0
    assert capsys.readouterr().out.strip() == (f"rag-learn {version('rag-learning-assistant')}")


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
        self.removed_document_ids: list[UUID] = []
        self.replacements: list[tuple[UUID, Path]] = []

    def add_document(self, path: Path) -> IndexedDocument:
        self.paths.append(path)
        return self.document

    def remove_document(self, document_id: UUID) -> IndexedDocument:
        self.removed_document_ids.append(document_id)
        return self.document

    def replace_document(
        self,
        document_id: UUID,
        path: Path,
    ) -> IndexedDocument:
        self.replacements.append((document_id, path))
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
    monkeypatch.setattr(
        commands,
        "build_pdf_extractor",
        lambda: FakePdfExtractor(document),
    )

    assert cli.main(["extract", str(pdf)]) == 0
    assert json.loads(capsys.readouterr().out) == {
        "source": "course.pdf",
        "pages_without_machine_readable_text": [],
        "pages": [
            {
                "number": 1,
                "source": "course.pdf",
                "text": "Lesson",
                "has_machine_readable_text": True,
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
    monkeypatch.setattr(
        commands,
        "build_pdf_extractor",
        lambda: FakePdfExtractor(document),
    )

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
    monkeypatch.setattr(
        commands,
        "build_pdf_extractor",
        lambda: FakePdfExtractor(document),
    )

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


def test_cli_removes_document_and_outputs_json(
    monkeypatch,
    tmp_path: Path,
    capsys,
) -> None:
    document_id = UUID("12345678-1234-5678-1234-567812345678")
    index_directory = tmp_path / "learning-library"
    index_directory.mkdir()
    (index_directory / "metadata.sqlite3").touch()
    document = IndexedDocument(
        id=document_id,
        source="python-book.pdf",
        content_sha256="a" * 64,
        page_count=10,
        chunk_count=25,
    )
    library_service = FakeLibraryService(document)

    monkeypatch.setattr(
        commands,
        "build_library_service",
        lambda chunker, index_directory: library_service,
    )

    result = cli.main(
        [
            "remove",
            str(document_id),
            "--index-dir",
            str(index_directory),
        ]
    )

    assert result == 0
    assert library_service.removed_document_ids == [document_id]
    assert json.loads(capsys.readouterr().out) == {
        "index_directory": str(index_directory),
        "removed_document": {
            "id": str(document_id),
            "source": "python-book.pdf",
            "content_sha256": "a" * 64,
            "page_count": 10,
            "chunk_count": 25,
        },
    }


def test_cli_remove_reports_unknown_document(
    monkeypatch,
    tmp_path: Path,
    capsys,
) -> None:
    document_id = UUID("aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa")
    index_directory = tmp_path / "learning-library"
    index_directory.mkdir()
    (index_directory / "metadata.sqlite3").touch()

    def raise_not_found(document_id: UUID) -> IndexedDocument:
        raise DocumentNotFoundError(f"Document does not exist: {document_id}")

    document = IndexedDocument(
        id=document_id,
        source="unused.pdf",
        content_sha256="a" * 64,
        page_count=1,
        chunk_count=1,
    )
    library_service = FakeLibraryService(document)
    monkeypatch.setattr(
        library_service,
        "remove_document",
        raise_not_found,
    )
    monkeypatch.setattr(
        commands,
        "build_library_service",
        lambda chunker, index_directory: library_service,
    )

    with pytest.raises(SystemExit) as exc_info:
        cli.main(
            [
                "remove",
                str(document_id),
                "--index-dir",
                str(index_directory),
            ]
        )

    assert exc_info.value.code == 2
    assert f"Document does not exist: {document_id}" in capsys.readouterr().err


def test_cli_replaces_document_and_outputs_json(
    monkeypatch,
    tmp_path: Path,
    capsys,
) -> None:
    document_id = UUID("12345678-1234-5678-1234-567812345678")
    replacement_pdf = tmp_path / "new-book.pdf"
    replacement_pdf.touch()
    index_directory = tmp_path / "learning-library"
    index_directory.mkdir()
    (index_directory / "metadata.sqlite3").touch()
    (index_directory / "vectors.faiss").touch()
    document = IndexedDocument(
        id=document_id,
        source="new-book.pdf",
        content_sha256="b" * 64,
        page_count=8,
        chunk_count=20,
    )
    library_service = FakeLibraryService(document)

    monkeypatch.setattr(
        commands,
        "build_library_service",
        lambda chunker, index_directory: library_service,
    )

    result = cli.main(
        [
            "replace",
            str(document_id),
            str(replacement_pdf),
            "--index-dir",
            str(index_directory),
        ]
    )

    assert result == 0
    assert library_service.replacements == [(document_id, replacement_pdf)]
    assert json.loads(capsys.readouterr().out) == {
        "index_directory": str(index_directory),
        "replaced_document": {
            "id": str(document_id),
            "source": "new-book.pdf",
            "content_sha256": "b" * 64,
            "page_count": 8,
            "chunk_count": 20,
        },
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


def test_parser_accepts_remove_command() -> None:
    document_id = UUID("12345678-1234-5678-1234-567812345678")

    args = build_parser().parse_args(
        [
            "remove",
            str(document_id),
            "--index-dir",
            "local-data/indexes/learning",
        ]
    )

    assert args.command == "remove"
    assert args.document_id == document_id
    assert args.index_dir == Path("local-data/indexes/learning")
    assert not hasattr(args, "max_chars")
    assert not hasattr(args, "overlap_chars")


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

    def fail_if_library_is_built(*args, **kwargs):
        raise AssertionError("Library service must not be built for an invalid index directory")

    monkeypatch.setattr(
        commands,
        "build_library_service",
        fail_if_library_is_built,
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


def test_index_accepts_gui_library_metadata(tmp_path: Path) -> None:
    index_directory = tmp_path / "course-index"
    index_directory.mkdir()
    (index_directory / "metadata.sqlite3").touch()
    (index_directory / "library.json").write_text(
        '{"id":"11111111-1111-1111-1111-111111111111","name":"Python"}',
        encoding="utf-8",
    )

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


def test_parser_accepts_replace_command() -> None:
    document_id = UUID("12345678-1234-5678-1234-567812345678")

    args = build_parser().parse_args(
        [
            "replace",
            str(document_id),
            "new-book.pdf",
            "--index-dir",
            "local-data/indexes/learning",
            "--max-chars",
            "800",
            "--overlap-chars",
            "100",
        ]
    )

    assert args.command == "replace"
    assert args.document_id == document_id
    assert args.pdf == Path("new-book.pdf")
    assert args.index_dir == Path("local-data/indexes/learning")
    assert args.max_chars == 800
    assert args.overlap_chars == 100
