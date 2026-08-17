# 0023: Evaluate written study answers automatically

## Status

Accepted

## Context

Interactive study sessions initially asked the learner to classify their own
answer as `again`, `hard`, `good`, or `easy`. Self-rating is transparent and
cheap, but it permits recognition and optimism to replace demonstrated recall.
The session already requires a written answer and stores trusted expected
answers and citations, so it has enough bounded material for automatic feedback.

The language model must not control source identity or scheduling directly. Its
output is probabilistic, can be malformed, and may follow instructions embedded
in document text unless that boundary is made explicit.

## Decision

The interactive `study` command always requires a non-blank written answer. An
`AnswerEvaluationService` sends the question, learner answer, expected answer,
and trusted citation excerpts to a narrow answer-evaluation generator. All
supplied text is marked as untrusted data, prior knowledge is forbidden, and the
model returns only a strict JSON evaluation.

The validated evaluation contains one of `incorrect`, `partially_correct`, or
`correct`, a finite score from zero to one, constructive feedback, missing
concepts, and the exact prompt references used. A correct result cannot contain
missing concepts; every non-correct result must identify at least one. Duplicate
concepts and prompt references are rejected.

The application, not the model, maps verdicts to review ratings:

- `incorrect` becomes `again`;
- `partially_correct` becomes `hard`;
- `correct` becomes `good`.

One correct answer never becomes `easy`. A future rule may use repeated stable
performance and response timing, but the model may not return a rating itself.
`ReviewScheduler` remains the only component that calculates the next due time.

The Hugging Face adapter uses versioned evaluation system and JSON-repair
prompts. It permits exactly one format-only repair attempt. The repair prompt
must preserve verdict, score, feedback, and missing concepts and may not add
facts, sources, citations, or scheduling decisions.

After evaluation, the CLI reveals the expected answer and trusted sources,
prints the verdict, score, feedback, missing concepts, derived rating, and next
due time. Evaluation data and complete prompt provenance are persisted inside
the immutable `StudyAttempt`. The nullable SQLite column is added automatically
so attempts created before this decision remain readable with no evaluation.

The explicit `review-record` command remains available for administration and
manual schedule correction, but the interactive `study` workflow no longer asks
the learner for a self-rating.

## Consequences

- Active recall is demonstrated through text before any solution is shown.
- Feedback is inspectable and grounded in the same trusted material as the
  question.
- Scheduling remains deterministic even though evaluation is model-generated.
- Malformed responses cannot partially advance progress or write an attempt.
- Small local models may still misjudge semantically equivalent answers; a
  future manual override can correct such cases without making model output a
  source of truth.
- Each interactive answer now incurs one local generation call and therefore
  has higher latency and GPU cost than self-rating.
