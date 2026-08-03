# 0007: Persist retrieval with FAISS and SQLite

## Status

Accepted

## Context

Re-embedding a PDF for every search is slow and prevents reuse across application runs.
The application needs efficient vector similarity search while retaining chunk text, source references, and enough model information to detect incompatible indices.

FAISS specializes in dense-vector search but does not serve as a general document metadata database.
SQLite provides transactional local metadata storage without an additional service.

## Decision

Use an exact FAISS inner-product index for normalized embeddings and SQLite for chunks and index metadata.
Persist model name, pinned revision, and embedding dimension with each index.
Map FAISS numeric IDs to SQLite chunk rows.

Write document embeddings in batches so FAISS is loaded and written once per indexing operation.
Expose indexing and searching as separate CLI commands.
Require indexing targets to be new or empty until document replacement semantics exist.

Use CPU FAISS on native Windows while allowing Sentence Transformers to generate embeddings on CUDA.

## Consequences

- Searches can reuse vectors without reopening or re-embedding the PDF.
- Source text and page metadata remain queryable through SQLite.
- Changing the embedding model or revision requires a new index.
- FAISS and SQLite form one logical store and require future consistency and recovery work for interrupted writes.
- Updating or combining indexed documents requires explicit document identity semantics in a later milestone.
