"""Render the published architecture diagrams as SVG files."""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DIAGRAM_DIRECTORY = PROJECT_ROOT / "docs" / "diagrams"
DIAGRAM_SOURCES = (
    *sorted((PROJECT_ROOT / "docs").glob("*-overview.puml")),
    *sorted(
        source for source in DIAGRAM_DIRECTORY.glob("*.puml") if not source.name.startswith("_")
    ),
)


def build_parser() -> argparse.ArgumentParser:
    """Build command-line arguments for selecting a PlantUML installation."""

    parser = argparse.ArgumentParser(description="Render architecture diagrams as SVG files")
    parser.add_argument(
        "--plantuml-jar",
        type=Path,
        help="Path to plantuml.jar (overrides PLANTUML_JAR and automatic discovery)",
    )
    return parser


def vscode_plantuml_jars() -> list[Path]:
    """Return PlantUML jars installed by the VS Code extension, newest first."""

    extension_root = Path.home() / ".vscode" / "extensions"
    return sorted(
        extension_root.glob("jebbs.plantuml-*/plantuml.jar"),
        reverse=True,
    )


def jar_command(explicit_jar: Path | None) -> list[str] | None:
    """Build a Java command when a PlantUML jar and Java are available."""

    configured_jar = explicit_jar
    if configured_jar is None and os.environ.get("PLANTUML_JAR"):
        configured_jar = Path(os.environ["PLANTUML_JAR"])

    candidates = [configured_jar] if configured_jar is not None else vscode_plantuml_jars()
    if not candidates:
        return None

    jar = candidates[0].expanduser().resolve()
    if not jar.is_file():
        raise RuntimeError(f"PlantUML jar does not exist: {jar}")

    java = shutil.which("java")
    if java is None:
        raise RuntimeError("Java is required to run plantuml.jar but was not found in PATH")
    return [java, "-jar", str(jar)]


def plantuml_command(explicit_jar: Path | None) -> list[str]:
    """Select a PlantUML executable or jar command."""

    if explicit_jar is not None or os.environ.get("PLANTUML_JAR"):
        command = jar_command(explicit_jar)
        assert command is not None
        return command

    executable = shutil.which("plantuml")
    if executable is not None:
        return [executable]

    command = jar_command(None)
    if command is not None:
        return command

    raise RuntimeError(
        "PlantUML was not found. Install the VS Code PlantUML extension, put "
        "plantuml in PATH, or pass --plantuml-jar PATH."
    )


def main() -> None:
    """Render every published diagram next to its PlantUML source."""

    args = build_parser().parse_args()
    command = [
        *plantuml_command(args.plantuml_jar),
        "-tsvg",
        *[str(source) for source in DIAGRAM_SOURCES],
    ]
    print(f"> {' '.join(command)}", flush=True)
    subprocess.run(command, check=True, cwd=PROJECT_ROOT)
    print(f"\nRendered {len(DIAGRAM_SOURCES)} architecture diagrams.")


if __name__ == "__main__":
    main()
