# 0004: Separate query and document embeddings

- Status: Accepted
- Date: 2026-08-03

## Context

Asymmetric retrieval compares short questions with longer knowledge passages.
Models such as E5 are trained with different prefixes for these two roles.
A single generic `embed(texts)` method cannot express that distinction reliably.

## Decision

Define separate `embed_documents(texts)` and `embed_query(text)` operations in the `Embedder` protocol.
Provider adapters own role-specific transformations.

## Consequences

- E5 query and passage prefixes are applied consistently.
- The application service remains independent of model-specific prompt conventions.
- Symmetric embedding providers must implement both operations, even if their internal behavior is identical.
