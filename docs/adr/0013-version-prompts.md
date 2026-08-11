# 0013: Version generation prompts explicitly

## Status

Accepted

## Context

Generated answers and summaries depend not only on source data and model revisions but also on the exact instructions sent to the model. Git history can show when source code changed, but a runtime result cannot identify its prompt configuration from a commit repository alone.

Future evaluation, persistence, and caching need to distinguish results created with different instructions. Storing complete prompt text with every result would duplicate data and could expose source-independent instructions unnecessarily. A manually managed version alone is also insufficient because prompt text could change without its version being incremented.

## Decision

Represent each stable instruction as an immutable `PromptTemplate` with a descriptive name, positive integer version, exact text, and an automatically calculated SHA-256 fingerprint of its UTF-8 text.

Keep prompt definitions with their owning responsibility: structured-output and repair prompts remain in the Hugging Face adapter, question-answering instructions remain in the question-answering application service, and map and reduce instructions remain in the summarization service. Dynamic questions, chunks, and partial summaries are runtime inputs and do not contribute to the template fingerprint.

Expose a compact `PromptReference` containing only name, version, and fingerprint. Generated domain results aggregate the references actually used, preserve first-use order, and remove duplicates created by repeated map calls. CLI JSON exposes these references without including complete prompt text.

Record the JSON repair prompt only when a repair generation actually occurs. A result produced without model generation, such as an answer with no relevant search contexts, has no prompt references.

## Consequences

- Runtime results identify the exact stable instructions used to create them.
- Human-readable versions support release notes and evaluation comparisons.
- SHA-256 detects accidental text changes when a version is not incremented.
- Prompt text remains owned by the component that defines its behavior.
- CLI consumers receive an additional `prompts` field in generated answer and summary JSON.
- Future cache identities can combine prompt references with model revision, generation settings, and source-document identity.
- Changing whitespace in stable prompt text changes its fingerprint and therefore requires the same review discipline as a semantic prompt change.
