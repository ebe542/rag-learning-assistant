# 0022: Record interactive study sessions atomically

## Status

Accepted

## Context

Persisted question banks and current review schedules allow questions to be
selected at the right time, but they do not preserve what the learner actually
answered. Updating only the current schedule loses the evidence needed to
inspect learning history, compare later scheduling policies, or add grounded
answer feedback.

An interactive session also creates two related persistent facts: an immutable
attempt and the question's new current schedule. Writing them independently can
leave the library inconsistent if one database operation succeeds and the other
fails.

## Decision

Each completed question creates an immutable `StudyAttempt`. It records a UUID,
the exact document and question-bank identity, question number and snapshot,
learner answer, expected answer, trusted citations, self-rating, timezone-aware
answer timestamp, and the resulting `QuestionProgress` snapshot.

Attempt identity is append-only. Repeating an identical write is idempotent, but
reusing an attempt UUID for different content is rejected. Histories are queried
for one exact document, bank, and question in chronological order.

Review calculation and persistence are separated. `ReviewService.prepare_review`
calculates a new schedule without writing it, while the existing `record_review`
operation remains available for explicit schedule-only commands. Interactive
sessions use the preparation path and then persist the attempt and current
progress together on one SQLite connection and transaction. A failure therefore
rolls back both changes.

The `study` command selects exactly one highest-priority due question. It hides
the expected answer until the learner has entered a non-blank answer, then shows
the trusted sources and asks for `again`, `hard`, `good`, or `easy`. Invalid
interactive input is requested again instead of terminating the session.

Study attempts are derived from a document and its exact question bank. Removing
or successfully replacing that document deletes its attempt history through the
same derived-data lifecycle as summaries, question banks, and progress. Failed
document mutation preserves the history.

## Consequences

- Learner answers and schedule transitions survive application restarts.
- Current progress and immutable history cannot diverge through a partial SQLite
  write.
- Stored question, expected-answer, citation, and progress snapshots keep old
  attempts inspectable even if scheduling code later changes.
- Self-rating remains the only quality judgment; automatic answer feedback is a
  separate future concern.
- Attempt storage initializes the progress schema it participates in, which
  temporarily duplicates that table definition until explicit database
  migrations are introduced.
- The workflow remains single-user and intentionally stores no user or tenant
  identifier.
