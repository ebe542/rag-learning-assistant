import json
from pathlib import Path

from rag_learning_assistant import cli
from rag_learning_assistant.documents import Document, Page


def test_cli_outputs_machine_readable_json(monkeypatch, tmp_path: Path, capsys) -> None:
    pdf = tmp_path / "course.pdf"
    pdf.touch()
    document = Document("course.pdf", (Page(1, "Lesson", "course.pdf"),))
    monkeypatch.setattr(cli.PdfExtractor, "extract", lambda self, path: document)

    assert cli.main([str(pdf)]) == 0
    assert json.loads(capsys.readouterr().out) == {
        "source": "course.pdf",
        "pages": [{"number": 1, "source": "course.pdf", "text": "Lesson"}],
    }
