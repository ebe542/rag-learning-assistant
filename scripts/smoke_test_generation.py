"""Manual GPU smoke test for the local Hugging Face generator."""

import sys
from pathlib import Path

import torch
from dotenv import load_dotenv

from rag_learning_assistant.generation import (
    HuggingFaceTextGenerator,
)

EXPECTED_CITATIONS = (1,)

PROMPT = """
Question: What is two plus two?

Contexts:
<context number="1">
[1] source: example.txt, page 1, chunk 0
Two plus two equals four.
</context>
""".strip()


def main() -> int:
    """Run one real generation and report GPU usage."""

    dotenv_path = Path(__file__).resolve().parents[1] / ".env"

    try:
        # A local token improves Hub rate limits, but authentication is not a
        # prerequisite because the pinned model is publicly accessible.
        load_dotenv(dotenv_path=dotenv_path)
    except Exception as exc:
        print(
            f"Warning: could not load {dotenv_path}: {exc}",
            file=sys.stderr,
        )

    if not torch.cuda.is_available():
        print(
            "Smoke test requires a CUDA-capable GPU.",
            file=sys.stderr,
        )
        return 1

    # Reset peak statistics immediately before lazy model loading so the
    # reported value includes both model weights and answer generation.
    torch.cuda.reset_peak_memory_stats()

    try:
        generator = HuggingFaceTextGenerator(
            max_new_tokens=128,
        )
        result = generator.generate(PROMPT)
    except Exception as exc:
        print(
            f"Generation smoke test failed: {exc}",
            file=sys.stderr,
        )
        return 1

    print(f"GPU: {torch.cuda.get_device_name()}")
    print(f"Text: {result.text}")
    print(f"Citations: {result.citation_numbers}")
    print(f"Peak GPU MB: {torch.cuda.max_memory_allocated() / 1024**2:.1f}")

    # Exact answer wording may vary between model/runtime versions. The stable
    # contract is that the model returns the supplied supporting context.
    if result.citation_numbers != EXPECTED_CITATIONS:
        print(
            "Smoke test failed: expected citation (1,).",
            file=sys.stderr,
        )
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
