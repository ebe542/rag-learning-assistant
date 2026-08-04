# 0008: Identify documents with UUIDs and content hashes

## Status

Accepted

## Context

A persistent index must distinguish documents from chunks and support more than one PDF.
Filenames are not reliable identities because different files may share a name and identical content may be renamed.
Using only a content hash as the identity would change the document identity whenever its contents change.

## Decision

Represent every registered library document as an immutable `IndexedDocument` with a UUID, source name, SHA-256 content hash, page count, and chunk count.
Use the UUID as the stable document identity and the SHA-256 hash for duplicate-content detection.

Assign the UUID before chunk indexing and persist it with every resulting chunk.
Reject duplicate content before PDF extraction and embedding generation.
Allow a persistent library to contain multiple documents and expose its registered documents through the CLI.

## Consequences

- Renaming an unchanged file does not bypass duplicate detection.
- A future document update can retain its UUID while storing a new content hash.
- Chunks can be filtered or removed by document UUID in a later milestone.
- Indices created before this decision retain nullable document IDs and require explicit reindexing or migration to become managed library documents.
- Registration and vector persistence still need recovery semantics for interrupted multi-store writes.
