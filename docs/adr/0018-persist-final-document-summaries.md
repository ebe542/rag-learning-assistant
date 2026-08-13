# 0018: Persist final document summaries

## Status

Accepted

## Context

Document-wide summarization can require several expensive GPU generations. The
existing summary cache stores validated Map results so interrupted work can resume,
but it does not preserve the completed user-facing summary. Repeating the same
command therefore still performs the Reduce phase, and a one-batch document can
repeat its final generation unnecessarily.

A stored final result is safe to reuse only when the document and every influential
generation setting still match. The existing `GenerationIdentity` already combines
the document content hash, pinned model revision, prompt references, Map and Reduce
token limits, and batch character budget into a stable fingerprint.

Citation numbers alone are insufficient for durable output. They identify contexts
inside one generation request but do not independently preserve source, page, chunk,
or excerpt data needed to inspect the result later.

## Decision

Persist completed document summaries in the library's existing `metadata.sqlite3`.
Use a dedicated `document_summaries` table keyed by document UUID and generation
identity fingerprint. Store the final text, complete trusted citations, and prompt
references. Keep the Map-batch cache and final-summary repository as separate
responsibilities even though they share one SQLite database.

Load document metadata and construct the current generation identity before looking
up a final result. An exact repository hit may skip chunk loading and every model
call. A changed document hash, model revision, prompt version, generation limit, or
batch budget produces a different key and therefore cannot reuse the old result.

Make ordinary writes immutable and idempotent: writing identical content again is a
no-op, while conflicting content under the same key is rejected. Expose `--force`
as explicit authorization to regenerate and replace a final result. Forced runs
bypass both final-result reads and Map-cache reads and writes, ensuring that every
model phase is executed afresh without corrupting resumable Map state.

Persist only after all Map, Reduce, citation, and domain-model validation succeeds.

## Consequences

- Repeating `summarize` with an unchanged configuration returns immediately from
  SQLite after a small metadata and identity lookup.
- Completed summaries remain available after process restart.
- Multiple configuration-specific results for one document can coexist.
- Full citations make cached results directly inspectable without reconstructing
  the original prompt context numbering.
- Final summaries, Map batches, library metadata, and vector mappings remain in one
  SQLite file but use separate tables and interfaces.
- `--force` is intentionally more expensive and does not refresh the Map cache; a
  later normal run still uses the existing resumable state if no final result exists.
- Removing or replacing documents will eventually require explicit lifecycle rules
  for associated historical summaries. Identity checks prevent stale reuse in the
  meantime, but storage cleanup is a separate concern.
