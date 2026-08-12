# 0016: Configure Map and Reduce generation limits separately

## Status

Accepted

## Context

Document summarization uses the same local model for two different tasks. Map
calls create compact partial summaries from source-heavy prompts, while the Reduce
call creates the final answer from those partial summaries. A single shared output
limit cannot express these different needs.

Benchmarks with the redistributable fixture showed that a 192-token shared limit
could end during the first Map response, leaving invalid JSON even after the one
permitted repair attempt. Merely increasing the shared limit would also permit
every Map call to become unnecessarily verbose and would increase runtime. Loading
two independent generators would duplicate a model that already nearly fills the
available GPU memory.

After introducing a concise Map prompt, a real fixture run still truncated Map JSON
at 128 tokens. The same prompt completed both Map batches at 192 tokens. This makes
192 a measured reliability baseline for the current model rather than an arbitrary
increase in output length.

The limits also influence persisted results. If either one changes without changing
the generation identity, cached batches could be reused under a configuration that
the runtime result no longer accurately describes.

## Decision

Use one generator and allow callers to override `max_new_tokens` for each generation
call. Configure document summarization with separate positive limits:

- 192 generated tokens for each Map call;
- 384 generated tokens for the Reduce call.

The larger Reduce budget is based on the stable benchmark fixture. A 256-token
run produced truncated JSON after validation began requiring the complete Map
citation union; 384 tokens produced valid JSON and preserved citations 1–20.

Keep the generator's constructor value as a default for workflows that do not need
phase-specific behavior. Apply the same per-call limit to the original generation
and its JSON repair attempt.

Version the Map prompt as version 2 and require at most 80 words containing only the
most important supported claims. The word target leaves room within the token limit
for JSON syntax and citation numbers, while runtime parsing and citation validation
remain authoritative.

Require the final reduction to use evidence from every partial summary. The Reduce
prompt expresses that coverage requirement and application validation rejects a
result whose citations omit an entire Map batch.

Expose the limits as `--max-map-new-tokens` and
`--max-reduce-new-tokens`. Include both values in the canonical
`GenerationIdentity` fingerprint so configuration changes cannot collide in the
summary cache.

## Consequences

- Map and Reduce output budgets can evolve independently.
- One Hugging Face pipeline remains loaded and shared across all summary phases.
- Short Map results reduce intermediate context and discourage truncated JSON.
- Reduce retains a larger budget for the final grounded answer.
- Changing either limit or the Map prompt creates a different cache identity.
- The previous `--max-new-tokens` Summarization option is removed before a stable
  CLI release rather than maintained as an ambiguous compatibility alias.
- Token limits remain upper bounds rather than word counts or guarantees of valid
  output; strict parsing and one repair attempt still handle model non-compliance.
