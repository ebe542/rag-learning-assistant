# 0015: Resume interrupted document summaries

## Status

Accepted

## Context

Document-wide summaries use multiple map generations followed by one reduction. A
large document can require many long-running GPU calls. If the process is stopped
after several successful batches, starting every map generation again wastes time
and energy and makes experimentation unnecessarily expensive.

Reusing a partial result is safe only when its document content, model revision,
prompt configuration, generation limits, and batch plan still match. A failed or
malformed generation must not become persistent state. Cache data also belongs to
the same local library as the chunks from which it was created.

## Decision

Persist each validated map result in a `summary_batches` table in the library's
existing `metadata.sqlite3` database. Address it by the stable generation identity
fingerprint and the one-based batch number. Store the original context-number range
with every entry so a changed batch plan is detected before reuse.

Treat writes as immutable and idempotent. Saving the same result again succeeds,
but different content for an existing key is rejected rather than overwritten.
Validate citations before saving and apply the same validation to cached results.

Configure the cache only together with a generation-identity factory. The CLI
identity includes the pinned model name and revision, all map, reduce, structured
output, and repair prompt references, both generation limits, and the indexed
document's content SHA-256. Cache hits skip the expensive map call and its progress
message. The final reduction is deliberately regenerated because this milestone
caches independently completed map work, which is the dominant resumability need.

## Consequences

- An interrupted summary continues with the first missing map batch.
- Changing source content or generation configuration selects a different cache
  identity instead of silently reusing incompatible output.
- Changes to batch planning are detected through stored context ranges.
- Cached data remains local to and lifecycle-aligned with its document library.
- Invalid model output cannot poison later resume attempts.
- The database can retain entries for obsolete identities; explicit cache cleanup
  can be introduced when storage growth becomes relevant.
- The reduction still runs on every invocation, even when all map batches are
  cached.
