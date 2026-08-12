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
