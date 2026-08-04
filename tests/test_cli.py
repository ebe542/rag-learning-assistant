import json
from pathlib import Path

import pytest

from rag_learning_assistant import cli
from rag_learning_assistant.chunking import Chunk
from rag_learning_assistant.ingestion import Document, Page
from rag_learning_assistant.interfaces.cli import commands, entrypoint
from rag_learning_assistant.interfaces.cli.parser import build_parser
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


def test_parser_accepts_index_command() -> None:
    args = build_parser().parse_args(
        [
            "index",
            "book.pdf",
            "--index-dir",
            ".rag-index/book",
        ]
    )

    assert args.command == "index"
    assert args.pdf == Path("book.pdf")
    assert args.index_dir == Path(".rag-index/book")


def test_cli_index_persists_document(
    monkeypatch,
    tmp_path: Path,
    capsys,
) -> None:
    pdf = tmp_path / "course.pdf"
    pdf.touch()
    index_directory = tmp_path / "course-index"
    document = Document(
        "course.pdf",
        (Page(1, "Python functions", "course.pdf"),),
    )
    search_service = FakeDocumentSearchService([])

    monkeypatch.setattr(
        entrypoint.PdfExtractor,
        "extract",
        lambda self, path: document,
    )
    monkeypatch.setattr(
        commands,
        "build_persistent_document_search",
        lambda chunker, index_dir: search_service,
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
    assert search_service.indexed_documents == [document]
    assert json.loads(capsys.readouterr().out) == {
        "source": "course.pdf",
        "index_directory": str(index_directory),
        "chunks_indexed": 0,
    }


def test_index_rejects_non_empty_index_directory(
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
    assert "index directory must be empty" in capsys.readouterr().err


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
