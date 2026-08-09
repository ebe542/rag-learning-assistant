# 0011: Summarize complete documents with map-reduce

## Status

Accepted

## Context

Similarity search deliberately returns only the chunks most relevant to a query. That behavior is suitable for question answering, but it cannot provide a representative summary of an entire document because relevant sections may never be retrieved.

Long documents also exceed the practical context budget of the text-generation model. A document-wide summary therefore needs bounded processing without losing the relationship between generated claims and the original source chunks.

Model-provided source metadata cannot be trusted. The application must derive citations from persistent chunk records and reject citation numbers that do not belong to the material used during generation.

## Decision

Summarize a document by its library UUID and read all of its stored chunks in their original order. Verify that the number of stored chunks matches the document catalog before generation starts.

Use a map-reduce process for documents that exceed a configurable character budget. The map phase summarizes consecutive chunk batches and assigns document-wide context numbers. The reduce phase combines those partial summaries while retaining only the original context numbers supported by the map results. A document that fits into one batch skips the unnecessary reduce call.

Treat chunks and partial summaries as untrusted source material in generation prompts. Reconstruct every returned citation from stored chunk metadata instead of accepting source, page, index, or excerpt values from the model.

Expose the capability through a `summarize` CLI command and emit machine-readable JSON. Do not persist generated summaries in this milestone because they depend on model and prompt configuration that does not yet have a versioned cache identity.

## Consequences

- Every stored section of a document contributes to the summarization process.
- Memory and model context use remain bounded for long documents.
- Additional batches require additional model calls and increase execution time.
- The reduce phase can compress details present in individual batch summaries.
- Citations remain traceable to original chunks across both generation phases.
- Incomplete chunk storage and unsupported model citations fail explicitly instead of producing an apparently valid summary.
- Summary persistence can be introduced later with explicit model, prompt, and document-version invalidation rules.
