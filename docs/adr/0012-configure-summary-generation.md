# 0012: Make summary generation limits configurable

## Status

Accepted

## Context

Document-wide summarization performs multiple model calls for long documents. The source batch size controls how many map calls are required, while the generated-token limit bounds each partial and final response. Fixed values prevent operators from balancing runtime, memory use, and summary detail for different documents and hardware.

The local Qwen model can also produce factually usable content with malformed JSON. Accepting or heuristically rewriting malformed responses would weaken the structured generation contract and could alter citation data.

Long-running manual checks need visible progress and timing information. Diagnostic output must not corrupt the machine-readable JSON written by CLI commands.

## Decision

Expose positive `max_new_tokens` and `max_batch_chars` settings for document summarization while retaining conservative defaults. Pass both values explicitly from the CLI through dependency construction to the generator and summarization service.

Report map and reduce progress through an optional application callback. The CLI adapter writes progress to standard error and keeps summary JSON on standard output.

When a model response fails strict structured parsing, make exactly one additional generation request that asks the model to repair only the JSON representation. The repair instruction forbids adding or removing factual claims and citation numbers. Reject the result if the second response remains invalid.

Maintain a manual GPU smoke test using a small real indexed document. Report elapsed time and peak allocated GPU memory, handle user cancellation cleanly, and keep the full-book workload outside the smoke test.

## Consequences

- Summary runtime and output budget can be tuned without source changes.
- Conservative defaults remain suitable for machines with limited GPU memory.
- Progress is visible during model calls without invalidating JSON output.
- One malformed response no longer necessarily discards all completed map work in the current run.
- A repair can add one extra model call, but repeated malformed output fails deterministically.
- Character budgets are only an approximation of tokenizer context usage and require future token-aware batching.
- The full-document workflow remains expensive and does not yet persist partial summaries for restart or reuse.
