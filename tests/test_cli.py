import json
from pathlib import Path

import pytest

from rag_learning_assistant import cli
from rag_learning_assistant.chunking import Chunk
from rag_learning_assistant.ingestion import Document, Page
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
    monkeypatch.setattr(cli.PdfExtractor, "extract", lambda self, path: document)

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
    monkeypatch.setattr(cli.PdfExtractor, "extract", lambda self, path: document)

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
    monkeypatch.setattr(cli.PdfExtractor, "extract", lambda self, path: document)

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
    args = cli.build_parser().parse_args(
        [
            "search",
            "book.pdf",
            "What are Python functions?",
            "--limit",
            "3",
        ]
    )

    assert args.command == "search"
    assert args.pdf == Path("book.pdf")
    assert args.query == "What are Python functions?"
    assert args.limit == 3


def test_cli_search_outputs_ranked_results(
    monkeypatch,
    tmp_path: Path,
    capsys,
) -> None:
    pdf = tmp_path / "course.pdf"
    pdf.touch()
    document = Document(
        "course.pdf",
        (Page(4, "Python functions", "course.pdf"),),
    )
    chunk = Chunk(
        text="Python functions",
        source="course.pdf",
        page_number=4,
        index=0,
    )
    search_service = FakeDocumentSearchService([SearchResult(chunk=chunk, score=0.91)])

    monkeypatch.setattr(
        cli.PdfExtractor,
        "extract",
        lambda self, path: document,
    )
    monkeypatch.setattr(
        cli,
        "build_document_search",
        lambda chunker: search_service,
        raising=False,
    )

    result = cli.main(
        [
            "search",
            str(pdf),
            "How do Python functions work?",
            "--limit",
            "3",
        ]
    )

    assert result == 0
    assert search_service.indexed_documents == [document]
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
    pdf = tmp_path / "course.pdf"
    pdf.touch()

    with pytest.raises(SystemExit) as exc_info:
        cli.main(
            [
                "search",
                str(pdf),
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
    pdf = tmp_path / "course.pdf"
    pdf.touch()

    with pytest.raises(SystemExit) as exc_info:
        cli.main(
            [
                "search",
                str(pdf),
                query,
            ]
        )

    assert exc_info.value.code == 2
    assert "must not be blank" in capsys.readouterr().err
