# 0025: Select study sessions by learning-package name

## Status

Accepted

## Context

Learning packages give prepared material a stable user-facing name, but the
interactive `study` command still required a document UUID and question-bank
identity fingerprint. Those values preserve exact provenance, yet requiring a
learner to copy them from diagnostic output prevents the package abstraction
from becoming the normal product workflow.

The existing `StudySessionService` already owns due-question selection, written
answer evaluation, scheduling, and attempt persistence. Reimplementing those
responsibilities for packages would create two study workflows that could drift.

## Decision

Introduce `LearningPackageStudyService` as a thin application facade. It resolves
a case-insensitive package name, requires the package to be ready with an active
question-bank identity, and delegates both due-question selection and answer
recording to the existing study-session service.

Expose the facade through:

```text
rag-learn study --library LIBRARY --package PACKAGE_NAME
```

Keep the positional directory, document UUID, and question-bank fingerprint form
temporarily available for diagnostics and exact-version automation. Reject
partial or mixed selections instead of guessing which identity the user meant.

## Consequences

- Learners can begin a session using the same name used to prepare and list a
  package.
- UUIDs and fingerprints remain authoritative internally without being required
  in the normal workflow.
- Package readiness is checked before any question or answer operation.
- Study behavior, evaluation, scheduling, and persistence remain implemented in
  one existing service.
- CLI compatibility adds temporary argument-validation complexity until the
  technical form can move to a dedicated administrative interface.
