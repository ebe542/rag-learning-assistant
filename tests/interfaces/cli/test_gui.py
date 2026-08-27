from rag_learning_assistant.interfaces.cli import commands, entrypoint
from rag_learning_assistant.interfaces.cli.parser import build_parser


def test_parser_accepts_gui_options() -> None:
    args = build_parser().parse_args(["gui", "--port", "9000", "--no-browser"])

    assert args.command == "gui"
    assert args.port == 9000
    assert args.no_browser is True


def test_entrypoint_dispatches_gui_without_opening_browser(monkeypatch) -> None:
    calls: list[tuple[int, bool]] = []
    monkeypatch.setattr(
        commands,
        "run_gui",
        lambda port, open_browser: calls.append((port, open_browser)) or 0,
    )

    assert entrypoint.main(["gui", "--port", "9000", "--no-browser"]) == 0
    assert calls == [(9000, False)]
