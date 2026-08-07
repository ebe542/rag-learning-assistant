import sys
from types import SimpleNamespace
from typing import Any

import pytest

from rag_learning_assistant.generation.huggingface import (
    HuggingFaceTextGenerator,
)


class StaticPipeline:
    def __init__(
        self,
        output: list[dict[str, object]],
    ) -> None:
        self.output = output

    def __call__(
        self,
        messages: list[dict[str, str]],
        **options: Any,
    ) -> list[dict[str, object]]:
        return self.output


class RecordingPipeline:
    def __init__(self, response_text: str) -> None:
        self.response_text = response_text
        self.calls: list[tuple[list[dict[str, str]], dict[str, Any]]] = []

    def __call__(
        self,
        messages: list[dict[str, str]],
        **options: Any,
    ) -> list[dict[str, object]]:
        self.calls.append((messages, options))
        return [
            {
                "generated_text": [
                    *messages,
                    {
                        "role": "assistant",
                        "content": self.response_text,
                    },
                ]
            }
        ]


def test_generate_uses_chat_messages_and_parses_response() -> None:
    pipeline = RecordingPipeline(
        """
        {
          "text": "A function groups reusable instructions.",
          "citation_numbers": [1]
        }
        """
    )
    generator = HuggingFaceTextGenerator(
        pipeline=pipeline,
        max_new_tokens=256,
    )

    result = generator.generate("Question and retrieved contexts")

    assert result.text == "A function groups reusable instructions."
    assert result.citation_numbers == (1,)
    assert len(pipeline.calls) == 1

    messages, options = pipeline.calls[0]

    assert messages[0]["role"] == "system"
    assert "valid JSON" in messages[0]["content"]
    assert '"citation_numbers"' in messages[0]["content"]
    assert messages[1] == {
        "role": "user",
        "content": "Question and retrieved contexts",
    }
    assert options == {
        "max_new_tokens": 256,
        "do_sample": False,
        "clean_up_tokenization_spaces": False,
        "tokenizer_encode_kwargs": {
            "enable_thinking": False,
        },
    }


@pytest.mark.parametrize("max_new_tokens", [0, -1])
def test_generator_requires_positive_token_limit(
    max_new_tokens: int,
) -> None:
    with pytest.raises(
        ValueError,
        match="max_new_tokens must be positive",
    ):
        HuggingFaceTextGenerator(
            pipeline=RecordingPipeline("{}"),
            max_new_tokens=max_new_tokens,
        )


@pytest.mark.parametrize(
    "output",
    [
        [],
        [{}],
        [{"generated_text": "not a conversation"}],
        [{"generated_text": []}],
        [
            {
                "generated_text": [
                    {
                        "role": "assistant",
                        "content": 42,
                    }
                ]
            }
        ],
    ],
)
def test_generate_rejects_invalid_pipeline_output(
    output: list[dict[str, object]],
) -> None:
    generator = HuggingFaceTextGenerator(
        pipeline=StaticPipeline(output),
    )

    with pytest.raises(
        ValueError,
        match="Hugging Face pipeline returned an invalid chat response",
    ):
        generator.generate("Question and contexts")


def test_pipeline_is_loaded_lazily_and_reused(
    monkeypatch,
) -> None:
    pipeline = RecordingPipeline(
        """
        {
          "text": "Grounded answer.",
          "citation_numbers": [1]
        }
        """
    )
    load_calls: list[str] = []
    generator = HuggingFaceTextGenerator()

    def load_pipeline() -> RecordingPipeline:
        load_calls.append("loaded")
        return pipeline

    monkeypatch.setattr(
        generator,
        "_load_pipeline",
        load_pipeline,
    )

    assert load_calls == []

    generator.generate("First prompt")
    generator.generate("Second prompt")
    assert load_calls == ["loaded"]


def test_pipeline_loading_uses_pinned_model_revision(
    monkeypatch,
) -> None:
    pipeline = RecordingPipeline(
        """
        {
          "text": "Grounded answer.",
          "citation_numbers": []
        }
        """
    )
    loaded_with: dict[str, str] = {}

    def load_pipeline(
        task: str,
        *,
        model: str,
        revision: str,
        dtype: str,
        device_map: str,
    ) -> RecordingPipeline:
        loaded_with.update(
            {
                "task": task,
                "model": model,
                "revision": revision,
                "dtype": dtype,
                "device_map": device_map,
            }
        )
        return pipeline

    fake_transformers = SimpleNamespace(
        pipeline=load_pipeline,
    )
    monkeypatch.setitem(
        sys.modules,
        "transformers",
        fake_transformers,
    )

    generator = HuggingFaceTextGenerator(
        model_name="Qwen/Qwen3-1.7B",
        model_revision=("70d244cc86ccca08cf5af4e1e306ecf908b1ad5e"),
    )

    generator.generate("Question and contexts")

    assert loaded_with == {
        "task": "text-generation",
        "model": "Qwen/Qwen3-1.7B",
        "revision": ("70d244cc86ccca08cf5af4e1e306ecf908b1ad5e"),
        "dtype": "auto",
        "device_map": "auto",
    }
