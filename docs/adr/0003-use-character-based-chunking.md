# 0003: Start with character-based chunking

- Status: Accepted
- Date: 2026-08-03

## Context

Token counts depend on a specific tokenizer and model.
The first MVP needs predictable, fast, dependency-free chunking before a final generation model has been selected.

## Decision

Measure maximum chunk size and overlap in characters.
Prefer paragraph and word boundaries, hard-split overlong words, and apply overlap only inside paragraphs that must be divided.

## Consequences

- Chunking is deterministic and independent of ML libraries.
- Character limits do not guarantee a specific token count.
- A tokenizer-aware strategy may supersede this decision when model context limits become operationally important.
