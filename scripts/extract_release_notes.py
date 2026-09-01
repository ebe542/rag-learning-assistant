"""Extract one version's release notes from the project changelog."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CHANGELOG = PROJECT_ROOT / "CHANGELOG.md"


def extract_release_notes(changelog: str, version: str) -> str:
    """Return exactly one non-empty dated changelog section."""

    heading = re.compile(rf"## \[{re.escape(version)}\] - \d{{4}}-\d{{2}}-\d{{2}}")
    lines = changelog.splitlines()
    matches = [index for index, line in enumerate(lines) if heading.fullmatch(line)]
    if len(matches) != 1:
        raise ValueError(
            f"Expected exactly one dated changelog section for {version}, found {len(matches)}"
        )

    start = matches[0] + 1
    end = next(
        (index for index in range(start, len(lines)) if lines[index].startswith("## ")),
        len(lines),
    )
    notes = "\n".join(lines[start:end]).strip()
    if not notes:
        raise ValueError(f"Changelog section for {version} is empty")
    return f"{notes}\n"


def build_parser() -> argparse.ArgumentParser:
    """Build the release-note extraction command."""

    parser = argparse.ArgumentParser(
        description="Extract one dated version section from CHANGELOG.md",
    )
    parser.add_argument("version", help="Package version without the leading v")
    parser.add_argument(
        "--changelog",
        type=Path,
        default=DEFAULT_CHANGELOG,
        help="Changelog source (default: project CHANGELOG.md)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Markdown file receiving the extracted release notes",
    )
    return parser


def main() -> None:
    """Extract the requested changelog section into a release-notes file."""

    args = build_parser().parse_args()
    notes = extract_release_notes(
        args.changelog.read_text(encoding="utf-8"),
        args.version,
    )
    args.output.write_text(notes, encoding="utf-8")


if __name__ == "__main__":
    main()
