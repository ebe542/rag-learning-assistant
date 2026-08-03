# 0005: Use multilingual E5 as the first embedding model

- Status: Accepted
- Date: 2026-08-03

## Context

The learning assistant needs local semantic retrieval for German and English documents.
The initial model should run on a developer machine, have broad ecosystem support, and provide a credible baseline before larger models are benchmarked.

## Decision

Use `intfloat/multilingual-e5-small` through Sentence Transformers as the first real embedding provider.
Pin model revision `614241f622f53c4eeff9890bdc4f31cfecc418b3` and request normalized embeddings.
Sentence Transformers is an optional dependency so the project core remains lightweight.

## Consequences

- Embeddings have 384 dimensions and support multilingual retrieval.
- Inputs require `query: ` and `passage: ` prefixes.
- The model truncates inputs at its supported token limit; chunk configuration must be evaluated against that constraint.
- Changing the model or revision requires rebuilding persisted vector indices.
- Larger candidates such as BGE-M3 should be compared using a project-specific retrieval evaluation set before adoption.
