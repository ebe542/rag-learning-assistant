"""Reproduce one GitHub CI Python job in a clean local environment."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CI_EXTRAS = "dev,gui,storage"


def environment_python(environment: Path) -> Path:
    """Return the virtual-environment interpreter on every supported platform."""

    if os.name == "nt":
        return environment / "Scripts" / "python.exe"
    return environment / "bin" / "python"


def run(command: list[str], *, python_path: Path | None = None) -> None:
    """Run one visible CI setup or verification command."""

    display = [str(python_path) if item == "{python}" else item for item in command]
    print(f"\n> {' '.join(display)}", flush=True)
    subprocess.run(display, check=True, cwd=PROJECT_ROOT)


def build_parser() -> argparse.ArgumentParser:
    """Build arguments for selecting the Python used to create the clean env."""

    parser = argparse.ArgumentParser(
        description="Run the GitHub CI quality gate in a fresh local virtual environment",
    )
    parser.add_argument(
        "--python",
        default=sys.executable,
        help="Python interpreter used to create the environment (default: active Python)",
    )
    return parser


def main() -> None:
    """Install only CI dependencies and run the shared quality gate."""

    args = build_parser().parse_args()
    with tempfile.TemporaryDirectory(prefix="rag-learning-assistant-ci-") as directory:
        environment = Path(directory) / "venv"
        run([args.python, "-m", "venv", str(environment)])
        python = environment_python(environment)
        run(
            [
                "{python}",
                "-m",
                "pip",
                "install",
                "--disable-pip-version-check",
                "-e",
                f".[{CI_EXTRAS}]",
            ],
            python_path=python,
        )
        # CI includes the lightweight GUI and storage adapters exercised by the
        # suite, but intentionally omits the large local-model dependencies.
        # Importing project tooling must therefore not require Torch during
        # collection.
        run(
            [
                "{python}",
                "-c",
                (
                    "import importlib.util; "
                    "assert importlib.util.find_spec('torch') is None; "
                    "import scripts.benchmark_summarization"
                ),
            ],
            python_path=python,
        )
        run(
            ["{python}", "scripts/check_milestone.py"],
            python_path=python,
        )

    print("\nClean CI environment checks passed.")


if __name__ == "__main__":
    main()
