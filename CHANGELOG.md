# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project uses [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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

[0.3.0a1]: https://github.com/ebe542/rag-learning-assistant/compare/v0.2.0...v0.3.0a1
[0.2.0]: https://github.com/ebe542/rag-learning-assistant/releases/tag/v0.2.0
