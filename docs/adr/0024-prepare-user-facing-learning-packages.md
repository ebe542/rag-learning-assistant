# 0024: Prepare user-facing learning packages

## Status

Accepted

## Context

The system can already index documents, create persisted summaries, generate
grounded question banks, and schedule study sessions. Using those capabilities
directly requires document UUIDs and generation fingerprints, however. Those
identities are necessary for reproducibility but are implementation details
rather than useful product concepts for a learner.

Long-running local model operations can also fail after earlier phases have
completed. Repeating successful indexing or generation work would waste time
and GPU resources.

## Decision

Introduce a persistent `LearningPackage` as the user-facing projection over one
document and its active summary and question-bank identities. A package has a
case-insensitive unique name and records the last successful preparation state:
`indexed`, `summarized`, or `ready`.

`LearningPackageService` coordinates the existing document, summarization, and
question-bank services. It persists a checkpoint after every expensive phase
and resumes from that checkpoint when preparation is repeated. The package
references versioned results instead of duplicating their contents.

Expose `prepare` and `package-list` as product-level CLI commands. Internal UUIDs
and fingerprints remain in JSON for diagnostics and automation, but normal
product workflows select material by package name.

For map-reduce summaries, Reduce must use supported evidence from every Map
section. The application retains the complete validated Map citation union;
the model is not required to repeat every global citation number.

## Consequences

- A PDF can become ready-to-study material through one resumable command.
- Completed work is reused after interruption and repeated commands are cheap.
- Product interfaces no longer need to expose orchestration details.
- One active learning package is currently allowed per document.
- Removing or replacing a document removes its package projection with the
  other derived data.
- Package names become the stable user-facing selector while technical
  identities continue to provide provenance and cache safety.
