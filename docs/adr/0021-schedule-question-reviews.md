# 0021: Persist and schedule self-rated question reviews

## Status

Accepted

## Context

Persisted question banks provide reusable learning material, but they do not
record whether a learner has answered a question or when it should be shown
again. Repeating every question equally would ignore demonstrated knowledge and
make larger banks increasingly inefficient.

Review state must remain attached to one exact question-bank version. A question
number alone is not stable across documents or generated banks, and progress
must not silently move to different source material after a document changes.
Time values also need an unambiguous representation across machines and daylight
saving transitions.

## Decision

Each question review is identified by document UUID, question-bank identity
fingerprint, and question number. Its current state stores a successful
repetition count, interval in days, ease factor, next due timestamp, and optional
last-review timestamp. Timestamps must be timezone-aware and the CLI records new
events in UTC.

Learners rate their own answer as `again`, `hard`, `good`, or `easy`. The first
algorithm deliberately uses a small, inspectable variant inspired by SM-2:

- `again` resets the successful sequence, lowers ease, and schedules a retry in
  ten minutes;
- `hard` grows the interval conservatively and lowers ease;
- `good` schedules the first two successful reviews after one and six days,
  then multiplies the interval by ease;
- `easy` starts at four days, grows more quickly, and increases ease;
- ease never falls below 1.3.

The algorithm is treated as project policy rather than an exact SM-2
implementation. Its transitions are isolated in a pure scheduler that returns a
new immutable state. The application service separately validates the exact
persisted bank and question, loads prior progress, invokes the scheduler, and
persists the result.

Questions without progress are new and immediately due, but listing them does
not create database rows. Existing due reviews are ordered by oldest due date
before new questions, preventing accumulated reviews from being displaced by
new material. New questions retain their question-bank order.

SQLite stores only the current schedule for each question. Saving progress is an
intentional upsert because a review replaces the previous current state. Removing
or successfully replacing a document deletes its summaries, question banks, and
review progress through the shared derived-data lifecycle. Failed index changes
preserve all derived state.

The CLI exposes `review-due` for selecting work and `review-record` for storing a
self-rating. Answer text is not persisted in this milestone; later interactive
study sessions can add answer capture and automated feedback without coupling it
to the scheduling policy.

## Consequences

- Learners can resume a versioned question bank across application restarts.
- The scheduling policy is deterministic, testable, and replaceable without
  changing persistence or CLI coordination.
- Self-rating keeps the first workflow local and transparent but does not verify
  whether an answer was factually correct.
- Day-based intervals are easy to inspect, while the ten-minute retry is
  represented directly by `due_at` with an interval value of zero.
- Only current state is stored; review history and analytics require a later
  append-only event model.
- Regenerating a bank with the same identity assumes deterministic generation;
  explicit per-question identity may be added if stochastic generation is
  introduced.
