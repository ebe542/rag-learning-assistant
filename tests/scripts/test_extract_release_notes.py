import sys
from pathlib import Path

import pytest

from scripts import extract_release_notes

CHANGELOG = """# Changelog

## [0.5.0a1] - 2026-09-01

### Added

- adaptive release notes

### Fixed

- missing descriptions

## [0.4.0a1] - 2026-08-30

### Added

- earlier feature
"""


def test_extract_release_notes_returns_only_requested_version() -> None:
    notes = extract_release_notes.extract_release_notes(CHANGELOG, "0.5.0a1")

    assert (
        notes
        == """### Added

- adaptive release notes

### Fixed

- missing descriptions
"""
    )


@pytest.mark.parametrize(
    "changelog",
    [
        "# Changelog\n",
        """## [0.5.0a1] - 2026-09-01

- first

## [0.5.0a1] - 2026-09-02

- duplicate
""",
    ],
)
def test_extract_release_notes_requires_exactly_one_section(changelog: str) -> None:
    with pytest.raises(ValueError, match="exactly one dated changelog section"):
        extract_release_notes.extract_release_notes(changelog, "0.5.0a1")


def test_extract_release_notes_rejects_empty_section() -> None:
    changelog = """## [0.5.0a1] - 2026-09-01

## [0.4.0a1] - 2026-08-30

- earlier
"""

    with pytest.raises(ValueError, match="section for 0.5.0a1 is empty"):
        extract_release_notes.extract_release_notes(changelog, "0.5.0a1")


def test_main_writes_release_notes_file(tmp_path: Path, monkeypatch) -> None:
    changelog = tmp_path / "CHANGELOG.md"
    output = tmp_path / "release-notes.md"
    changelog.write_text(CHANGELOG, encoding="utf-8")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "extract_release_notes.py",
            "0.5.0a1",
            "--changelog",
            str(changelog),
            "--output",
            str(output),
        ],
    )

    extract_release_notes.main()

    assert output.read_text(encoding="utf-8") == extract_release_notes.extract_release_notes(
        CHANGELOG,
        "0.5.0a1",
    )
