# 0020: Generate persistent grounded question banks from summaries

## Status

Accepted

## Context

Study questions should be reusable learning material rather than transient model
output. Generating them directly from an entire document would repeat expensive
document-wide processing and make it harder to identify which source selection
produced a particular bank. A persisted grounded summary already provides a
validated, versioned representation of the document together with trusted
citations to the original chunks.

Question banks also outlive the generation process. Without an explicit identity
and lifecycle, results created with different models, prompts, limits, or summary
versions could be confused, and banks could remain accessible after their source
document changes.

## Decision

Question generation uses one explicitly selected persisted summary identity. The
summary text provides the overview used to formulate questions, while its stored
citations provide the only permitted evidence. Citation numbers returned by the
model are resolved back to those trusted citation objects; the model cannot
invent source metadata.

Each bank receives a deterministic identity derived from the model name and
revision, every available question-generation prompt, the requested question
count, the generation token limit, and the selected summary identity. Prompt
references attached to the stored result record which prompts were actually
used, including the repair prompt only when repair was necessary.

Generation output is parsed as a strict JSON schema. Questions must be nonblank,
unique, consecutively numbered, and grounded by at least one valid citation. One
format-repair attempt is allowed. Normal generation reuses an identical stored
bank; `--force` explicitly regenerates and replaces it.

Question banks are persisted in the library's SQLite database and exposed
through a separate read-only catalog. The CLI generates, lists, and shows banks
without conflating those operations. Removing or successfully replacing a
document deletes both its summaries and question banks through a general
derived-data cleanup interface. Failed index mutations preserve all derived data
for diagnosis.

## Consequences

- Question generation is fast relative to rereading and summarizing an entire
  document, but overview questions are limited to information preserved by the
  selected summary.
- Stored questions remain traceable to exact document passages and to all inputs
  that affect reproducibility.
- Different summary versions and generation configurations can coexist without
  being mistaken for one another.
- Listing and reading existing banks do not load the generation model.
- Document lifecycle coordination can clean up future derived learning material
  without adding another dedicated dependency to `LibraryService`.
- Detailed questions based directly on sections or chunks remain a separate
  future capability.
