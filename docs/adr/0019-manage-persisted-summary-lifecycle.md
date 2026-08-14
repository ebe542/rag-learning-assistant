# 0019: Manage persisted summary lifecycles with library documents

## Status

Accepted

## Context

Final document summaries are durable derived data. Once they are persisted, they
can outlive the source content from which they were generated unless document
management explicitly owns their lifecycle. Keeping such summaries after a
document is removed would expose information that no longer belongs to the
library. Keeping them after replacement would make old content appear current
even though the stable document UUID now identifies revised source material.

Users also need to inspect which generation identities are stored and retrieve
one exact result without invoking the model again. A missing document and a
known document without a matching summary are different conditions and should
remain distinguishable.

## Decision

Library removal and replacement coordinate cleanup of every persisted final
summary associated with the document UUID. Cleanup occurs only after the chunk
operation succeeds and before the catalog metadata is removed or updated. If
chunk removal is inconsistent or replacement indexing fails, existing summaries
remain visible so that the failure is not hidden behind partially deleted data.

The application exposes a separate read-only document-summary catalog. It first
validates that the document still belongs to the library, then lists all stored
generation identities or retrieves one exact identity. The CLI reflects this
separation with `summary-list` for compact version metadata and `summary-show`
for the full grounded result, citations, and prompt references.

Generation identity fingerprints remain the stable selectors for final summary
variants. CLI input accepts hexadecimal SHA-256 fingerprints and normalizes
their case before lookup.

## Consequences

- Removed or replaced library content cannot leave accessible final summaries
  behind.
- Failed document mutations preserve summaries and catalog metadata for
  diagnosis instead of concealing an inconsistent index.
- Users can inspect cached results without loading embedding or generation
  models.
- Listing remains compact, while full text and provenance require an explicit
  identity selection.
- The summary Map cache remains an implementation cache keyed by generation
  identity; this decision governs final user-visible summaries only.
- Document chunks, catalog metadata, and final summaries span storage concerns,
  so coordination reduces stale data but is not a single cross-store database
  transaction.
