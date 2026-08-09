"""Hugging Face adapter for local chat-based text generation."""

from typing import Any, Protocol, cast

from rag_learning_assistant.generation.models import (
    GenerationResult,
)
from rag_learning_assistant.generation.response_parser import (
    parse_generation_response,
)

SYSTEM_PROMPT = """
Return only valid JSON with exactly these fields:
{
  "text": "the grounded answer",
  "citation_numbers": [1, 2]
}
Use citation_numbers only for contexts that directly support the answer.
Do not wrap the JSON in Markdown code fences.
""".strip()
DEFAULT_MODEL_NAME = "Qwen/Qwen3-1.7B"
DEFAULT_MODEL_REVISION = "70d244cc86ccca08cf5af4e1e306ecf908b1ad5e"


class PipelineGenerationConfig(Protocol):
    """Mutable generation settings exposed by a Transformers pipeline."""

    max_length: int | None
    max_new_tokens: int | None
    do_sample: bool
    temperature: float | None
    top_p: float | None
    top_k: int | None


class ChatPipeline(Protocol):
    """Subset of the Transformers chat pipeline used by the adapter."""

    @property
    def generation_config(self) -> PipelineGenerationConfig:
        """Expose mutable generation settings without replacing the config object."""

        ...

    def __call__(
        self,
        messages: list[dict[str, str]],
        **options: Any,
    ) -> list[dict[str, object]]:
        """Generate a chat continuation."""

        ...


class HuggingFaceTextGenerator:
    """Generate structured answers through a local chat pipeline."""

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL_NAME,
        model_revision: str = DEFAULT_MODEL_REVISION,
        pipeline: ChatPipeline | None = None,
        max_new_tokens: int = 512,
    ) -> None:
        if max_new_tokens < 1:
            raise ValueError("max_new_tokens must be positive")

        self.model_name = model_name
        self.model_revision = model_revision
        self._pipeline = pipeline
        self.max_new_tokens = max_new_tokens

    def generate(self, prompt: str) -> GenerationResult:
        """Generate and parse one grounded answer."""

        messages = [
            {
                "role": "system",
                "content": SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": prompt,
            },
        ]
        outputs = self._get_pipeline()(
            messages,
            clean_up_tokenization_spaces=False,
            # Qwen3 enables reasoning by default. Disabling it prevents
            # <think> blocks from corrupting the strict JSON response.
            tokenizer_encode_kwargs={
                "enable_thinking": False,
            },
        )

        # Chat pipelines return the complete conversation. The final message is
        # the newly generated assistant response, not part of the input prompt.
        raw_response = self._extract_response(outputs)

        return parse_generation_response(raw_response)

    def _get_pipeline(self) -> ChatPipeline:
        """Load and configure the local model once."""

        if self._pipeline is None:
            self._pipeline = self._load_pipeline()

        # Transformers 5 expects generation settings in one place. Clearing
        # max_length avoids competing limits when max_new_tokens is configured.
        self._pipeline.generation_config.max_length = None
        self._pipeline.generation_config.max_new_tokens = self.max_new_tokens
        self._pipeline.generation_config.do_sample = False

        # Sampling controls are meaningless during deterministic greedy
        # decoding and cause Transformers to report an invalid configuration.
        self._pipeline.generation_config.temperature = 1.0
        self._pipeline.generation_config.top_p = 1.0
        self._pipeline.generation_config.top_k = 50

        return self._pipeline

    def _load_pipeline(self) -> ChatPipeline:
        """Load the pinned model through Transformers."""

        try:
            from transformers import pipeline
        except ImportError as exc:
            raise RuntimeError(
                "Transformers is not installed. Install the generation optional dependency."
            ) from exc

        loaded_pipeline = pipeline(
            "text-generation",
            model=self.model_name,
            revision=self.model_revision,
            dtype="auto",
            device_map="auto",
        )

        # Transformers exposes a broad pipeline union. This adapter narrows it
        # to the chat behavior validated by _extract_response at runtime.
        return cast(ChatPipeline, loaded_pipeline)

    @staticmethod
    def _extract_response(
        outputs: list[dict[str, object]],
    ) -> str:
        """Extract the assistant text from a chat pipeline response."""

        error_message = "Hugging Face pipeline returned an invalid chat response"

        if not outputs:
            raise ValueError(error_message)

        conversation = outputs[0].get("generated_text")

        if not isinstance(conversation, list) or not conversation:
            raise ValueError(error_message)

        assistant_message = conversation[-1]

        if not isinstance(assistant_message, dict):
            raise ValueError(error_message)

        if assistant_message.get("role") != "assistant":
            raise ValueError(error_message)

        content = assistant_message.get("content")

        if not isinstance(content, str):
            raise ValueError(error_message)

        return content
