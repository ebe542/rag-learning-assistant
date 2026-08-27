import tomllib
from pathlib import Path

import pytest

from scripts import check_release_package


def test_build_backend_is_bounded_to_twine_compatible_metadata() -> None:
    configuration = tomllib.loads(
        (check_release_package.PROJECT_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    )

    assert configuration["build-system"]["requires"] == ["hatchling>=1.27,<1.28"]


def test_project_version_reads_package_metadata() -> None:
    assert check_release_package.project_version() == "0.3.0a1"


def test_environment_python_uses_platform_layout(monkeypatch) -> None:
    environment = Path("clean-environment")

    monkeypatch.setattr(check_release_package.os, "name", "nt")
    assert check_release_package.environment_python(environment) == (
        environment / "Scripts" / "python.exe"
    )

    monkeypatch.setattr(check_release_package.os, "name", "posix")
    assert check_release_package.environment_python(environment) == environment / "bin" / "python"


def test_parser_uses_active_python_by_default() -> None:
    args = check_release_package.build_parser().parse_args([])

    assert args.python == check_release_package.sys.executable


def test_parser_accepts_explicit_python() -> None:
    args = check_release_package.build_parser().parse_args(["--python", "python3.12"])

    assert args.python == "python3.12"


@pytest.mark.parametrize(
    ("filename", "suffix"),
    [
        ("rag_learning_assistant-0.3.0a1-py3-none-any.whl", ".whl"),
        ("rag_learning_assistant-0.3.0a1.tar.gz", ".tar.gz"),
    ],
)
def test_find_distribution_returns_only_matching_file(
    tmp_path: Path,
    filename: str,
    suffix: str,
) -> None:
    distribution = tmp_path / filename
    distribution.write_bytes(b"package")

    assert check_release_package.find_distribution(tmp_path, suffix) == distribution


def test_find_distribution_rejects_missing_file(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="Expected exactly one .whl distribution, found 0"):
        check_release_package.find_distribution(tmp_path, ".whl")
