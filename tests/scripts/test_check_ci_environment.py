from pathlib import Path

from scripts import check_ci_environment


def test_environment_python_uses_platform_layout(monkeypatch) -> None:
    environment = Path("clean-environment")

    monkeypatch.setattr(check_ci_environment.os, "name", "nt")
    assert check_ci_environment.environment_python(environment) == (
        environment / "Scripts" / "python.exe"
    )

    monkeypatch.setattr(check_ci_environment.os, "name", "posix")
    assert check_ci_environment.environment_python(environment) == environment / "bin" / "python"


def test_parser_uses_active_python_by_default() -> None:
    args = check_ci_environment.build_parser().parse_args([])

    assert args.python == check_ci_environment.sys.executable


def test_parser_accepts_explicit_python() -> None:
    args = check_ci_environment.build_parser().parse_args(["--python", "python3.12"])

    assert args.python == "python3.12"
