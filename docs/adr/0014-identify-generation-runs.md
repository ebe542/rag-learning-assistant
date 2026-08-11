# 0014: Identify document generation configurations

## Status

Accepted

## Context

Resumable and cached document summaries must never reuse model output created from incompatible source data or generation settings. Prompt references identify the stable instructions, but they do not identify the model, generation limits, batching policy, or document content involved in a run.

A cache key must be deterministic across supported processes and platforms. Relying on Python object hashes, dictionary insertion order, or implicit serialization would make persisted entries unstable. Missing or duplicate configuration values would also create ambiguous identities.

## Decision

Represent one document-generation configuration as an immutable `GenerationIdentity`. Include the model name and pinned revision, an ordered and unique tuple of prompt references, the generated-token limit, the source batch character limit, and the document content SHA-256.

Validate that model fields are non-blank, limits are positive, at least one prompt is present, prompt references are unique, and the document hash is a canonical lowercase SHA-256 digest.

Serialize the complete configuration as canonical JSON with sorted keys, UTF-8 encoding, and fixed compact separators. Use the SHA-256 digest of that representation as the generation fingerprint. Lock the canonical representation with a known fingerprint test so changes require an explicit cache migration decision.

Keep this identity independent of SQLite and summary orchestration. Persistence will use it as a key in a subsequent milestone.

## Consequences

- Equal document-generation configurations produce the same stable fingerprint.
- Changes to source content, model configuration, prompts, token limits, or batch limits produce different cache identities.
- Cache entries cannot silently cross document revisions or generation policies.
- Prompt ordering remains significant because prompt roles and first-use order can affect generation behavior.
- Changing the canonical JSON representation invalidates existing fingerprints and requires a deliberate migration or cache rebuild.
- The identity currently models document summarization requirements; other generation workflows may later need additional input-specific identity fields.
