# 0001: Use a Python src layout

- Status: Accepted
- Date: 2026-08-03

## Context

Importing directly from a repository root can accidentally make tests pass against the working tree even when the installed package is incomplete.
The project also needs clear separation between package code, tests, and repository metadata.

## Decision

Place importable application code under `src/rag_learning_assistant` and configure Hatchling to package that directory.

## Consequences

- Tests exercise the installed package layout.
- Repository files cannot accidentally become importable modules.
- Editable installation is required for normal local development.
