"""Run the complete quality gate used to finish a development milestone."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def run(command: list[str]) -> None:
    """Run one check and stop immediately when it fails."""

    print(f"\n> {' '.join(command)}", flush=True)
    subprocess.run(command, check=True, cwd=PROJECT_ROOT)


def main() -> None:
    """Verify formatting, static checks, tests, coverage, and resource cleanup."""

    # Using the active interpreter keeps the check inside the selected virtual
    # environment on Windows, Linux, and macOS.
    python = sys.executable

    run([python, "-m", "ruff", "format", "--check", "."])
    run([python, "-m", "ruff", "check", "."])
    run(
        [
            python,
            "-m",
            "pytest",
            "--cov=rag_learning_assistant",
            "--cov-report=term-missing",
            "--cov-fail-under=90",
            "--basetemp=.pytest-tmp",
            "-p",
            "no:cacheprovider",
            "-W",
            "error::ResourceWarning",
            "-W",
            "error::pytest.PytestUnraisableExceptionWarning",
        ]
    )
    run(["git", "diff", "--check"])

    print("\nMilestone checks passed.")


if __name__ == "__main__":
    main()
