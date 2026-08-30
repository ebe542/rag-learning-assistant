from pathlib import Path

from rag_learning_assistant.interfaces.cli import commands, entrypoint
from rag_learning_assistant.interfaces.cli.parser import build_parser
from rag_learning_assistant.interfaces.web import server


def test_parser_accepts_gui_options() -> None:
    args = build_parser().parse_args(
        [
            "gui",
            "--library",
            "personal-library",
            "--port",
            "9000",
            "--no-browser",
        ]
    )

    assert args.command == "gui"
    assert args.library == Path("personal-library")
    assert args.port == 9000
    assert args.no_browser is True


def test_entrypoint_dispatches_gui_without_opening_browser(monkeypatch) -> None:
    calls: list[tuple[Path, int, bool]] = []
    monkeypatch.setattr(
        commands,
        "run_gui",
        lambda library_directory, port, open_browser: (
            calls.append((library_directory, port, open_browser)) or 0
        ),
    )

    assert (
        entrypoint.main(
            [
                "gui",
                "--library",
                "personal-library",
                "--port",
                "9000",
                "--no-browser",
            ]
        )
        == 0
    )
    assert calls == [(Path("personal-library"), 9000, False)]


def test_run_gui_supplies_lazy_package_service_factory(monkeypatch) -> None:
    calls = []
    package_service = object()
    monkeypatch.setattr(
        commands,
        "build_learning_package_service",
        lambda directory, progress: package_service,
    )

    def fake_run_server(**kwargs) -> None:
        calls.append(kwargs)

    monkeypatch.setattr(server, "run_server", fake_run_server)

    assert commands.run_gui(Path("personal-library"), 9000, open_browser=False) == 0

    assert calls[0]["library_directory"] == Path("personal-library")
    assert calls[0]["port"] == 9000
    assert calls[0]["open_browser"] is False
    factory = calls[0]["package_service_factory"]
    assert factory(Path("library-id"), lambda phase: None) is package_service
