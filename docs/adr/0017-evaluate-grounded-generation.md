# 0017: Evaluate grounded generation with versioned reference cases

## Status

Accepted

## Context

Prompt, retrieval, embedding, and model changes can alter generated answers even
when the application remains technically correct. Test success alone therefore
cannot show whether grounded-answer quality improved or regressed. Exact answer
text is unsuitable as a reference because multiple formulations can express the
same supported fact.

Citation-only evaluation is also insufficient. An answer can cite the expected
page while omitting or contradicting its important content. Conversely, semantic
grading by the same model under evaluation would add cost, non-determinism, and a
second prompt whose behavior would itself require evaluation.

The repository already contains a stable, redistributable ten-page benchmark PDF.
Its controlled content makes source pages and key ideas suitable for versioned
reference cases without depending on private learning documents.

## Decision

Store grounded-generation cases as versioned JSON beside the benchmark fixture.
Each case contains a stable ID, an exact question, expected source-page citations,
and optional expected concepts. Identify citations by source filename and one-based
page number rather than chunk index because chunk boundaries may legitimately
change independently of the source evidence.

Represent each concept with a stable name and one or more accepted phrases. Match
phrases as case-insensitive substrings of the generated answer. Treat a case as
passed only when every expected citation and concept is present and no unexpected
citation is returned.

Report citation recall, citation precision, concept recall, individual failures,
and the generated answer text as versioned JSON. Keep human-readable progress on
standard error so standard output remains machine-readable. Use the production
question-answering builder so evaluation exercises the same retrieval, prompting,
generation, and trusted-citation reconstruction as `rag-learn ask`.

Keep generated reports under `local-data` by default. The versioned case file is
the durable specification; hardware- and model-run output is evidence to inspect,
not a universal guarantee.

## Consequences

- Prompt and model experiments can be compared against stable source expectations.
- Reports distinguish retrieval or citation regressions from missing answer ideas.
- Answer text makes failed cases inspectable without reproducing an expensive run.
- Matching is deterministic, inexpensive, and requires no additional model.
- Accepted phrases must be curated when a correct paraphrase produces a false
  negative. Such additions must remain faithful to the named concept rather than
  merely copying one observed answer.
- Substring matching does not prove semantic correctness, detect negation, or
  establish that every claim is supported. The baseline is a regression signal,
  not a complete factuality evaluation.
- English reference phrases currently match the English synthetic fixture and
  questions. Multilingual cases require their own reviewed phrase alternatives.
