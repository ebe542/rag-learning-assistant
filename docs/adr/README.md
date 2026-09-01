# Architecture Decision Records

Architecture Decision Records (ADRs) capture decisions that have a lasting impact on the project.
Each record describes its context, decision, and consequences.

ADRs are immutable after acceptance.
If a decision changes, add a new ADR that supersedes the earlier record instead of rewriting history.

## Index

- [0001: Use a Python src layout](0001-use-src-layout.md)
- [0002: Preserve PDF page boundaries](0002-preserve-page-boundaries.md)
- [0003: Start with character-based chunking](0003-use-character-based-chunking.md)
- [0004: Separate query and document embeddings](0004-separate-query-and-document-embeddings.md)
- [0005: Use multilingual E5 as the first embedding model](0005-use-multilingual-e5-small.md)
- [0006: Introduce application services for workflow orchestration](0006-introduce-application-services.md)
- [0007: Persist retrieval with FAISS and SQLite](0007-persist-retrieval-with-faiss-and-sqlite.md)
- [0008: Identify documents with UUIDs and content hashes](0008-identify-library-documents.md)
- [0011: Summarize complete documents with map-reduce](0011-document-wide-summarization.md)
- [0012: Make summary generation limits configurable](0012-configure-summary-generation.md)
- [0013: Version generation prompts explicitly](0013-version-prompts.md)
- [0014: Identify document generation configurations](0014-identify-generation-runs.md)
- [0015: Resume interrupted document summaries](0015-resume-document-summaries.md)
- [0016: Configure Map and Reduce generation limits separately](0016-separate-map-and-reduce-generation-limits.md)
- [0017: Evaluate grounded generation with versioned reference cases](0017-evaluate-grounded-generation.md)
- [0018: Persist final document summaries](0018-persist-final-document-summaries.md)
- [0019: Manage persisted summary lifecycles with library documents](0019-manage-persisted-summary-lifecycle.md)
- [0020: Manage grounded question banks](0020-manage-grounded-question-banks.md)
- [0021: Persist and schedule self-rated question reviews](0021-schedule-question-reviews.md)
- [0022: Record interactive study sessions atomically](0022-record-interactive-study-sessions.md)
- [0023: Evaluate written study answers automatically](0023-evaluate-written-study-answers.md)
- [0024: Prepare user-facing learning packages](0024-prepare-user-facing-learning-packages.md)
- [0025: Select study sessions by learning-package name](0025-study-learning-packages-by-name.md)
- [0026: Report learning-package progress from persisted study data](0026-report-learning-package-progress.md)
- [0027: Generate question banks in resumable batches](0027-resume-question-bank-generation.md)
- [0028: Introduce a local web interface](0028-introduce-local-web-interface.md)
- [0029: Adapt JSON repair budgets for truncated model output](0029-adapt-json-repair-budgets.md)
