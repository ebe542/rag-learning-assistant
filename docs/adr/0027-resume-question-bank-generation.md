# 0027: Generate question banks in resumable batches

## Status

Accepted

## Context

Generating a complete question bank in one model call produces a large JSON
response. Real preparation runs with twenty questions took several minutes
without observable progress and could end with truncated JSON even after the
single format-repair attempt. An interruption discarded the whole generation
although some questions may already have been produced internally.

Increasing the output-token limit would make the single operation more
expensive without removing its all-or-nothing failure mode. Partial results can
only be reused safely when their model, prompts, source summary, token limit,
question count, and batch plan are identical.

## Decision

Generate question banks in batches of five questions by default. Include the
batch size in `QuestionBankIdentity`, so different plans receive different
SHA-256 fingerprints and never share cached output.

Persist every validated batch in the library's `metadata.sqlite3` database.
Address a batch by question-bank identity fingerprint and one-based batch
number, and store its exact global question-number range. Writes are immutable
and idempotent: identical retries succeed, while conflicting content is
rejected.

Validate citation numbers, prompt provenance, numbering, and normalized
question-text uniqueness before a new batch is cached. Supply earlier question
texts to later prompts so the model can avoid repetition. If a generated batch
still duplicates accepted or internally repeated text, make exactly one
semantic repair attempt. The versioned repair prompt treats the accepted and
rejected texts as forbidden questions. Validate the complete replacement batch
through the same boundary and cache only a valid result. Only after every batch
is present does the application assemble and persist the final question bank.

Report generated and reused batches through an application callback. The CLI
writes progress to standard error, preserving machine-readable JSON on standard
output. Normal runs resume from compatible cached batches. `force=True` ignores
intermediate cache reads and writes and replaces only the completed bank.

## Consequences

- Long question-generation runs expose real batch progress.
- Interrupting between model calls preserves every completed valid batch.
- A repeated command starts with the first missing batch instead of restarting
  the whole question bank.
- Smaller JSON responses reduce truncation risk but require more model calls.
- Previously generated questions increase later prompt size slightly.
- A duplicate batch costs at most one additional model call before it fails.
- Invalid citations, prompt provenance, numbering, or duplicate text cannot
  poison a new resume entry.
- Obsolete batch identities may remain in SQLite until an explicit cache-cleanup
  policy is introduced.
