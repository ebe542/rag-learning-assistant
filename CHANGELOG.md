# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project uses [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.5.0a1] - 2026-09-01

### Added

- deterministic German and English document-language detection
- per-package learning-language selection with same-language and translated summaries
- validation and targeted repair when a model returns a summary in the wrong language
- a public, configurable GUI worker smoke-test template for private PDF fixtures
- automated GitHub release publishing from verified annotated version tags

### Changed

- structured JSON generation now uses bounded adaptive repair budgets for summaries,
  question batches, and answer evaluations
- GUI package status polling replaces content only when its rendered state changes
- model failures write detailed bounded diagnostics to the private rotating log while
  the GUI shows a concise actionable message

### Fixed

- PDFs without machine-readable words are rejected before indexing and model processing
- invalid PDF control characters are removed while preserving normal tabs and line breaks
- translated summaries no longer silently remain in the document's source language
- longer multilingual JSON repairs no longer reuse the same insufficient token budget
- failed-package status and action buttons remain aligned beside long error messages

## [0.4.0a1] - 2026-08-30

### Added

- a loopback-only browser interface for the complete local learning workflow
- library creation, selection, renaming, and confirmed deletion
- PDF package uploads with persisted preparation requests and SHA-256 duplicate checks
- a serial background worker with resumable preparation phases, leases, and retry support
- live package-status updates without full-page reloads
- package summaries, study sessions, answer feedback, and progress views in the GUI
- collapsible package renaming and confirmed full package deletion
- linked PlantUML software, module, and SQLite data-model overviews

### Changed

- package uploads now redirect to the live package list instead of a stale validation page
- library directories use internal UUIDs independently of their display names
- user-facing review timestamps are shown in local time
- routine quality checks and release-package checks run in separate workflows

### Fixed

- duplicate PDFs renamed by the user are rejected before model processing starts
- failed preparations show concise explanations and can be retried or removed
- package preparation status and summaries remain current while background work runs
- SQLite resources are closed deterministically during tests and repository reopening

## [0.3.0a1] - 2026-08-27

### Added

- resumable, persisted question-bank generation with per-batch progress and timing
- user-facing `package-show` and `package-remove` commands
- platform-specific default learning-library directories with an environment override
- human-readable package and progress output with optional JSON output
- `doctor` diagnostics for Python, dependencies, library state, and CUDA
- an end-to-end alpha getting-started guide
- reproducible wheel and source-distribution validation in a clean environment
- separate GitHub Actions workflows for routine quality checks and release packages

### Changed

- simplified complete local installation through the `local` optional dependency group
- expanded package metadata and exposed the installed version through `rag-learn --version`
- hardened question-bank identity and duplicate replacement across resumable batches

### Fixed

- CLI failures now write diagnostic logs without exposing technical tracebacks
- MuPDF diagnostics no longer add terminal noise during normal operation
- interrupted summary reduction can resume without discarding valid completed work
- duplicate or malformed question batches can retain valid questions and refill shortfalls

## [0.2.0] - 2026-08-21

The `0.2.0` milestone established persistent summaries, grounded question banks,
spaced review, written-answer evaluation, resumable learning packages, and
learning-progress reporting.

[0.5.0a1]: https://github.com/ebe542/rag-learning-assistant/compare/v0.4.0a1...v0.5.0a1
[0.4.0a1]: https://github.com/ebe542/rag-learning-assistant/compare/v0.3.0a1...v0.4.0a1
[0.3.0a1]: https://github.com/ebe542/rag-learning-assistant/compare/v0.2.0...v0.3.0a1
[0.2.0]: https://github.com/ebe542/rag-learning-assistant/releases/tag/v0.2.0
