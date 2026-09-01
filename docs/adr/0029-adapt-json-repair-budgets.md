# 0029: Adapt JSON repair budgets for truncated model output

## Status

Accepted

Supersedes the fixed JSON-repair attempt count in ADRs 0012, 0020, and 0023.

## Context

The local model returns summaries, question batches, and answer evaluations as
strict JSON. A malformed representation can often be repaired without changing
its grounded content. Earlier decisions therefore allowed exactly one
format-only repair using a fixed output-token budget.

Real multilingual smoke tests showed that the initial response and its repair
can both end inside the same JSON string. German output may also require a
different number of tokens than comparable English output. Raising each task's
fixed limit after every failure makes normal generation unnecessarily expensive
and does not provide a consistent upper bound across structured-generation
features.

Malformed JSON is not always truncated. Repeating a schema error or an invalid
quoted string with progressively larger budgets adds latency without addressing
its cause. Recovery must distinguish an apparent output-boundary failure from
other invalid model responses.

## Decision

Keep strict parsing and the format-only repair prompts. Apply one shared,
bounded repair policy inside `HuggingFaceTextGenerator` for summaries, question
batches, and answer evaluations:

1. Generate once with the token budget selected by the caller.
2. If strict parsing fails, make one format-only repair request. Give it at
   least 512 tokens and normally twice the initial budget, without exceeding
   the configured repair ceiling.
3. If the repair remains invalid, inspect the JSON decoder failure. Only an
   unterminated string or a decoding failure at the response boundary qualifies
   as apparent truncation.
4. For apparent truncation, allow one final format-only repair with a doubled
   budget, capped at 1,024 tokens by default.
5. Reject immediately after a non-truncation repair failure, after reaching the
   ceiling, or after three total generation attempts.

The maximum repair budget and total attempt count are constructor settings. A
caller's initial budget may exceed the default repair ceiling; repair never
reduces that caller-selected budget. Successful repaired results record the
same versioned repair prompt reference regardless of whether one or two repair
calls were required.

Every terminal failure retains strict schema behavior and attaches bounded raw
responses to the private application log. The adapter does not heuristically
complete JSON, alter citations, or accept a partial domain object.

## Consequences

- Normal valid generations keep their original token budget and one model call.
- Ordinary malformed or schema-invalid JSON still receives at most one repair.
- Repeatedly truncated JSON can recover without changing task-specific limits.
- The default worst case is three model calls and a 1,024-token repair budget,
  so runtime and memory use remain bounded.
- All structured-generation features use the same recovery behavior.
- Detection is intentionally conservative; an unusual truncation that does not
  fail near the response boundary may still be rejected.
- Historical benchmark results and earlier ADRs remain valid evidence for the
  behavior they recorded, while this ADR replaces their fixed repair count.
