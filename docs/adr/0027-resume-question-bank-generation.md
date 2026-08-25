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

Generate question banks in batches of three questions by default. Include the
batch size in `QuestionBankIdentity`, so different plans receive different
SHA-256 fingerprints and never share cached output.

Keep expected answers to at most two sentences. Three questions fit the local
model's default 512-token response budget more reliably; real five-question
batches repeatedly truncated their final JSON object. The smaller batch trades
additional model calls for bounded output, narrower evidence, and safer resume
points.

Persist every validated batch in the library's `metadata.sqlite3` database.
Address a batch by question-bank identity fingerprint and one-based batch
number, and store its exact global question-number range. Writes are immutable
and idempotent: identical retries succeed, while conflicting content is
rejected.

Deterministically partition the persisted summary's citations into balanced,
contiguous evidence ranges for the batch plan. Keep the summary identity as a
generation input, but omit its complete generated text from question prompts.
Expose only the assigned contexts to each batch and accept only their citation
numbers. Use the same range for replacement generation. If there are more
batches than citations, reuse citations cyclically rather than creating an
evidence-free batch. Version the changed prompt so older all-context or
full-summary cache entries cannot be mixed with this strategy.

Validate citation numbers, prompt provenance, numbering, and normalized
question-text uniqueness before a new batch is cached. Supply earlier question
texts to later prompts so the model can avoid repetition. If a generated batch
still duplicates accepted or internally repeated text, retain its unique
candidates and generate every missing replacement in a separate semantic repair
call. Assign a distinct, balanced subset of the batch evidence to each call so
the model does not repeatedly receive the same broad topic. The versioned repair
prompt treats accepted earlier questions, accepted candidates, rejected
duplicates, and replacements already accepted in the sequence as forbidden.
Allow at most three attempts per replacement and add every failed duplicate to
the next attempt's forbidden list. Change the required focus across attempts
from a concrete example to a process and then a limitation or comparison. This
makes deterministic retries materially different. Validate every replacement through the same
boundary and cache only a complete valid batch. Treat the requested count as a
target upper bound: after bounded repair is exhausted, skip caching the partial
batch, assemble the available unique questions, persist the smaller valid bank,
and report its actual count to the user.

Report generated and reused batches through an application callback. The CLI
writes progress to standard error, preserving machine-readable JSON on standard
output. Normal runs resume from compatible cached batches. `force=True` ignores
intermediate cache reads and writes and replaces only the completed bank.
Measure each newly generated batch with a monotonic clock and report its elapsed
time only after validation and optional persistence. The measurement includes
all duplicate-replacement calls. Reused batches have no generated duration.

## Consequences

- Long question-generation runs expose real batch progress.
- Completed new batches expose comparable elapsed times without contaminating
  machine-readable output.
- Interrupting between model calls preserves every completed valid batch.
- A repeated command starts with the first missing batch instead of restarting
  the whole question bank.
- Smaller JSON responses reduce truncation risk but require more model calls.
- Separate evidence ranges encourage coverage of different document sections
  instead of repeatedly selecting the same easiest contexts.
- Previously generated questions increase later prompt size slightly.
- A duplicate batch costs up to three smaller model calls per missing question
  before accepting a shortfall. This is slower than a shared repair call but gives each
  replacement a narrower topic, an independent structured response budget, and
  bounded recovery from stochastic exact repetitions.
- Invalid citations, prompt provenance, numbering, or duplicate text cannot
  poison a new resume entry.
- Terminal JSON-repair failures attach bounded model output to the local
  technical traceback. This improves diagnosis but means the private
  rotating log can contain source-derived question text.
- Obsolete batch identities may remain in SQLite until an explicit cache-cleanup
  policy is introduced.
