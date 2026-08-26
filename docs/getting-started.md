# Getting started with the alpha

This guide runs the complete local learning workflow on Windows with Git Bash.
The alpha works with text-based PDF files and stores all learning data locally.
Scanned PDFs are not supported yet because OCR is not included.

## 1. Install the application

Open Git Bash in the repository directory and create a Python 3.13 virtual
environment:

```bash
py -3.13 -m venv .venv
source .venv/Scripts/activate
python -m pip install --upgrade pip
python -m pip install -e ".[embeddings,generation,storage]"
```

The local models are downloaded only when they are first needed. NVIDIA users
should follow the CUDA-specific PyTorch installation instructions in the main
[README](../README.md#nvidia-gpu-support-on-windows) before installing all
extras.

Activate the environment again after opening a new Git Bash terminal:

```bash
source .venv/Scripts/activate
```

## 2. Check the local environment

```bash
rag-learn doctor
```

The final line should be `Status: ready`. CUDA is optional; when it is
unavailable, generation uses the CPU and can take substantially longer.

The default library is stored in
`%LOCALAPPDATA%\rag-learning-assistant\library`. To use a different location for
the current terminal session:

```bash
export RAG_LEARN_LIBRARY="local-data/product-library"
```

Run `rag-learn doctor` again to confirm the selected path.

## 3. Prepare the first PDF

Choose a short, text-based PDF for the first run:

```bash
rag-learn prepare "books/python-basics.pdf" \
  --name "Python Basics" \
  --questions 5
```

Preparation indexes the document, creates a grounded summary, and generates a
question bank. The first run can take several minutes because it may download
and load the local models. Progress is written after each completed phase.

If preparation is interrupted, run the same command again. Completed phases
are persisted and reused when their configuration still matches.

## 4. Inspect the learning package

```bash
rag-learn package-list
rag-learn package-show --package "Python Basics"
```

The package is ready to study when its status is `ready` and both the summary
and questions are available.

Add `--json` to either command when structured output is needed for a script:

```bash
rag-learn package-show --package "Python Basics" --json
```

## 5. Start learning

```bash
rag-learn study --package "Python Basics"
```

Enter a written answer without looking at the expected answer. The local model
evaluates it against the trusted document passages, reports missing concepts,
and schedules the next review.

Repeat the command for the next due question. When no question is due, the
command reports that there is currently nothing to review.

## 6. Check progress

```bash
rag-learn progress --package "Python Basics"
```

The report shows answered and due questions, evaluated attempts, difficult
concepts, and the next review time. Use `--json` for machine-readable output.

## 7. Remove a package

First verify the exact package name:

```bash
rag-learn package-show --package "Python Basics"
```

Then remove it:

```bash
rag-learn package-remove --package "Python Basics"
```

Warning: this permanently removes the package, indexed source document,
generated summary and questions, review schedule, and complete study history.
The original PDF file is not deleted.

## Troubleshooting

Run the human-readable diagnosis first:

```bash
rag-learn doctor
```

For a support report that can be copied without reformatting:

```bash
rag-learn doctor --json
```

Common causes of failure:

- `rag-learn: command not found`: activate `.venv` again.
- A dependency is `missing`: repeat the installation command from step 1.
- The library is `incomplete`: select the intended library and verify that it
  contains both `metadata.sqlite3` and `vectors.faiss`.
- A PDF produces little or no text: it is probably scanned and requires OCR,
  which is not supported in this alpha.
- Generation uses the CPU unexpectedly: check the PyTorch CUDA installation
  described in the main README.

Technical identifiers and lower-level commands remain available for diagnosis
and automation in the main [README](../README.md).
