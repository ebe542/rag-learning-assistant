import json
from pathlib import Path

import pytest

from rag_learning_assistant import cli
from rag_learning_assistant.ingestion import Document, Page


def test_cli_outputs_machine_readable_json(monkeypatch, tmp_path: Path, capsys) -> None:
    pdf = tmp_path / "course.pdf"
    pdf.touch()
    document = Document("course.pdf", (Page(1, "Lesson", "course.pdf"),))
    monkeypatch.setattr(cli.PdfExtractor, "extract", lambda self, path: document)

    assert cli.main([str(pdf)]) == 0
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
                str(pdf),
                "--max-chars",
                "100",
                "--overlap-chars",
                "100",
            ]
        )

    assert exc_info.value.code == 2
    assert "overlap_chars must be smaller than max_chars" in capsys.readouterr().err
