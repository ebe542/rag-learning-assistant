# 0002: Preserve PDF page boundaries

- Status: Accepted
- Date: 2026-08-03

## Context

Grounded answers must cite the source document and page.
Chunks spanning multiple pages would require more complex source ranges and could make citations less precise.

## Decision

Extract and chunk every PDF page separately.
A chunk stores one source filename and one one-based page number.
Chunk indices still increase across the entire document.

## Consequences

- Every chunk has a simple and unambiguous citation.
- Content near page breaks cannot share one chunk.
- Retrieval may need adjacent-chunk expansion later to restore page-break context.
