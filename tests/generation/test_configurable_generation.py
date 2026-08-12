from typing import Any

import pytest

from rag_learning_assistant.generation.huggingface import HuggingFaceTextGenerator


class RecordingGenerationConfig:
    def __init__(self) -> None:
        self.max_length: int | None = 20
        self.max_new_tokens: int | None = None
        self.do_sample = True
        self.temperature: float | None = 0.7
        self.top_p: float | None = 0.8
        self.top_k: int | None = 20


class RecordingPipeline:
    def __init__(self) -> None:
        self.generation_config = RecordingGenerationConfig()
        self.observed_token_limits: list[int | None] = []

    def __call__(
        self,
        messages: list[dict[str, str]],
        **options: Any,
    ) -> list[dict[str, object]]:
        self.observed_token_limits.append(
            self.generation_config.max_new_tokens,
        )
        return [
            {
                "generated_text": [
                    *messages,
                    {
                        "role": "assistant",
                        "content": """
                        {
                          "text": "A grounded summary.",
                          "citation_numbers": [1]
                        }
                        """,
                    },
                ]
            }
        ]


def test_generate_can_override_default_token_limit_per_call() -> None:
    pipeline = RecordingPipeline()
    generator = HuggingFaceTextGenerator(
        pipeline=pipeline,
        max_new_tokens=512,
    )

    generator.generate(
        "Summarize one batch.",
        max_new_tokens=128,
    )
    generator.generate(
        "Combine partial summaries.",
        max_new_tokens=256,
    )

    assert pipeline.observed_token_limits == [128, 256]
    assert generator.max_new_tokens == 512


def test_generate_uses_default_token_limit_without_override() -> None:
    pipeline = RecordingPipeline()
    generator = HuggingFaceTextGenerator(
        pipeline=pipeline,
        max_new_tokens=192,
    )

    generator.generate("Answer a question.")

    assert pipeline.observed_token_limits == [192]


@pytest.mark.parametrize("max_new_tokens", [0, -1])
def test_generate_rejects_non_positive_token_override(
    max_new_tokens: int,
) -> None:
    pipeline = RecordingPipeline()
    generator = HuggingFaceTextGenerator(
        pipeline=pipeline,
    )

    with pytest.raises(
        ValueError,
        match="max_new_tokens must be positive",
    ):
        generator.generate(
            "Summarize one batch.",
            max_new_tokens=max_new_tokens,
        )

    # Invalid configuration must fail before loading or invoking the model.
    assert pipeline.observed_token_limits == []
