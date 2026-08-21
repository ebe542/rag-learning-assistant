# 0026: Report learning-package progress from persisted study data

## Status

Accepted

## Context

The product can prepare and list learning packages and conduct written-answer
study sessions by package name. Learners still cannot see whether a package is
new, actively studied, currently due, or repeatedly difficult without inspecting
technical SQLite data and reconstructing attempt histories manually.

Question progress and study attempts describe different facts. Progress stores
the current schedule for one question, while attempts form an immutable history.
One question can therefore have many attempts. Older attempts may also lack an
automatic answer evaluation and must not be silently classified from their
self-rating.

## Decision

Introduce an immutable `LearningProgressReport` and a read-only
`LearningProgressService`. Resolve a ready package by its case-insensitive name,
load its active question bank, and aggregate current progress and all attempts for
each question.

Report distinct total, answered, and due question counts separately from attempt
and verdict counts. Count attempts without an evaluation as `unclassified`.
Aggregate missing concepts from evaluated attempts and order them by descending
frequency with a deterministic name tie-breaker. Treat questions without a
schedule as immediately due. Expose the projection as machine-readable JSON:

```text
rag-learn progress --library LIBRARY --package PACKAGE_NAME
```

Calculate the report on demand without persisting another derived representation
or loading ML models.

## Consequences

- Learners can inspect useful progress using the same package name as preparation
  and study commands.
- Question and attempt rates have explicit, different denominators.
- Historical unevaluated attempts remain visible without inventing verdicts.
- Repeated reports are cheap and do not mutate learning state.
- The current implementation reads histories per question, which is simple and
  adequate for personal libraries but may need bulk repository methods for much
  larger question banks.
