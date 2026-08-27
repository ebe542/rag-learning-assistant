"""Build and verify installable release distributions in temporary storage."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tempfile
import tomllib
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DISTRIBUTION_NAME = "rag-learning-assistant"


def project_version() -> str:
    """Read the single release version declared in project metadata."""

    configuration = tomllib.loads((PROJECT_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    return str(configuration["project"]["version"])


def environment_python(environment: Path) -> Path:
    """Return the virtual-environment interpreter on every supported platform."""

    if os.name == "nt":
        return environment / "Scripts" / "python.exe"
    return environment / "bin" / "python"


def run(command: list[str]) -> None:
    """Run one visible packaging command from the project root."""

    print(f"\n> {' '.join(command)}", flush=True)
    subprocess.run(command, check=True, cwd=PROJECT_ROOT)


def find_distribution(distribution_directory: Path, suffix: str) -> Path:
    """Require exactly one newly built distribution with the requested suffix."""

    matches = sorted(distribution_directory.glob(f"*{suffix}"))
    if len(matches) != 1:
        raise RuntimeError(f"Expected exactly one {suffix} distribution, found {len(matches)}")
    return matches[0]


def build_parser() -> argparse.ArgumentParser:
    """Build arguments for selecting the clean-environment interpreter."""

    parser = argparse.ArgumentParser(
        description="Build, inspect, and install the release package in a clean environment",
    )
    parser.add_argument(
        "--python",
        default=sys.executable,
        help="Python interpreter used to create the clean environment (default: active Python)",
    )
    return parser


def main() -> None:
    """Build wheel and sdist, validate metadata, and smoke-test the wheel."""

    args = build_parser().parse_args()
    expected_version = project_version()
    with tempfile.TemporaryDirectory(prefix="rag-learning-assistant-release-") as directory:
        temporary_root = Path(directory)
        distribution_directory = temporary_root / "dist"
        environment = temporary_root / "venv"

        run(
            [
                sys.executable,
                "-m",
                "build",
                "--outdir",
                str(distribution_directory),
                ".",
            ]
        )
        wheel = find_distribution(distribution_directory, ".whl")
        source_distribution = find_distribution(distribution_directory, ".tar.gz")
        run(
            [
                sys.executable,
                "-m",
                "twine",
                "check",
                str(wheel),
                str(source_distribution),
            ]
        )

        run([args.python, "-m", "venv", str(environment)])
        python = environment_python(environment)
        run(
            [
                str(python),
                "-m",
                "pip",
                "install",
                "--disable-pip-version-check",
                str(wheel),
            ]
        )
        run(
            [
                str(python),
                "-c",
                (
                    "from importlib.metadata import version; "
                    f"assert version('{DISTRIBUTION_NAME}') == '{expected_version}'"
                ),
            ]
        )
        run([str(python), "-m", "rag_learning_assistant.cli", "--version"])
        run([str(python), "-m", "rag_learning_assistant.cli", "--help"])

    print("\nRelease package checks passed.")


if __name__ == "__main__":
    main()
