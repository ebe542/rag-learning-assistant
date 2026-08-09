import json
from pathlib import Path

import pytest

from rag_learning_assistant.generation import Citation, GroundedAnswer
from rag_learning_assistant.interfaces.cli import commands, entrypoint
from rag_learning_assistant.interfaces.cli.parser import (
    DEFAULT_RESULT_LIMIT,
    build_parser,
)


class RecordingQuestionAnsweringService:
    def __init__(self, answer: GroundedAnswer) -> None:
        self.result = answer
        self.calls: list[tuple[str, int]] = []

    def answer(self, question: str, limit: int) -> GroundedAnswer:
        self.calls.append((question, limit))
        return self.result


def test_run_ask_outputs_grounded_answer_as_json(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    question = "Was ist eine Python-Klasse?"
    answerer = RecordingQuestionAnsweringService(
        GroundedAnswer(
            question=question,
            text="Eine Klasse beschreibt Struktur und Verhalten von Objekten.",
            citations=(
                Citation(
                    number=2,
                    source="python-book.pdf",
                    page_number=42,
                    chunk_index=15,
                    excerpt="A class defines the structure and behavior of objects.",
                ),
            ),
        )
    )
    monkeypatch.setattr(
        commands,
        "build_question_answering_service",
        lambda index_directory: answerer,
        raising=False,
    )

    exit_code = commands.run_ask(
        index_directory=tmp_path,
        question=question,
        limit=3,
    )

    assert exit_code == 0
    assert answerer.calls == [(question, 3)]
    assert json.loads(capsys.readouterr().out) == {
        "question": question,
        "answer": "Eine Klasse beschreibt Struktur und Verhalten von Objekten.",
        "citations": [
            {
                "number": 2,
                "source": "python-book.pdf",
                "page_number": 42,
                "chunk_index": 15,
                "excerpt": "A class defines the structure and behavior of objects.",
            }
        ],
    }


def test_parser_accepts_ask_command() -> None:
    parser = build_parser()

    args = parser.parse_args(
        [
            "ask",
            "local-data/indexes/python-book",
            "Was ist ein Decorator?",
        ]
    )

    assert args.command == "ask"
    assert args.index_dir == Path("local-data/indexes/python-book")
    assert args.question == "Was ist ein Decorator?"
    assert args.limit == DEFAULT_RESULT_LIMIT


def test_parser_accepts_ask_result_limit() -> None:
    parser = build_parser()

    args = parser.parse_args(
        [
            "ask",
            "local-data/indexes/python-book",
            "Was ist ein Decorator?",
            "--limit",
            "5",
        ]
    )

    assert args.limit == 5


@pytest.mark.parametrize("limit", ["0", "-1"])
def test_parser_rejects_non_positive_ask_result_limit(limit: str) -> None:
    parser = build_parser()

    with pytest.raises(SystemExit):
        parser.parse_args(
            [
                "ask",
                "local-data/indexes/python-book",
                "Was ist ein Decorator?",
                "--limit",
                limit,
            ]
        )


def test_entrypoint_dispatches_ask_command(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    index_directory = tmp_path / "library"
    index_directory.mkdir()

    # The entry point validates the persistent index structure before building
    # expensive retrieval and generation dependencies.
    (index_directory / "vectors.faiss").touch()
    (index_directory / "metadata.sqlite3").touch()

    calls: list[tuple[Path, str, int]] = []

    def fake_run_ask(
        index_directory: Path,
        question: str,
        limit: int,
    ) -> int:
        calls.append((index_directory, question, limit))
        return 0

    monkeypatch.setattr(commands, "run_ask", fake_run_ask)

    exit_code = entrypoint.main(
        [
            "ask",
            str(index_directory),
            "Was ist eine Python-Klasse?",
            "--limit",
            "3",
        ]
    )

    assert exit_code == 0
    assert calls == [
        (
            index_directory,
            "Was ist eine Python-Klasse?",
            3,
        )
    ]


def test_entrypoint_rejects_incomplete_ask_index_before_loading_services(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    index_directory = tmp_path / "incomplete-library"
    index_directory.mkdir()
    (index_directory / "metadata.sqlite3").touch()

    def fail_if_called(**kwargs: object) -> int:
        raise AssertionError("Question answering must not start for an incomplete index")

    monkeypatch.setattr(commands, "run_ask", fail_if_called)

    with pytest.raises(SystemExit) as exc_info:
        entrypoint.main(
            [
                "ask",
                str(index_directory),
                "Was ist eine Python-Klasse?",
            ]
        )

    assert exc_info.value.code == 2
    assert "index directory is incomplete" in capsys.readouterr().err
