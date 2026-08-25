"""Hugging Face adapter for local chat-based text generation."""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING, Any, Protocol, cast

if TYPE_CHECKING:
    from rag_learning_assistant.learning.feedback import (
        AnswerEvaluation,
    )

from rag_learning_assistant.generation.feedback_parser import (
    parse_answer_evaluation,
)
from rag_learning_assistant.generation.models import (
    GenerationResult,
)
from rag_learning_assistant.generation.prompts import PromptTemplate
from rag_learning_assistant.generation.question_parser import (
    QuestionGenerationResult,
    parse_question_generation_response,
)
from rag_learning_assistant.generation.response_parser import (
    parse_generation_response,
)

SYSTEM_PROMPT = PromptTemplate(
    name="generation.system-json",
    version=2,
    text="""
Return only one valid JSON object with exactly these fields:
- "text": a string containing the actual answer requested by the user
- "citation_numbers": an array of integers identifying only contexts that
  directly support the answer

Never return placeholder text.
Never copy field descriptions into the response.
Do not wrap the JSON object in Markdown code fences.
""".strip(),
)

JSON_REPAIR_PROMPT = PromptTemplate(
    name="generation.json-repair",
    version=2,
    text="""
Your previous response did not match the required JSON format.
Preserve the actual answer content from the previous response.
Never replace it with placeholder text or a schema example.
Return the response again as valid JSON with exactly the fields "text" and
"citation_numbers".
Correct only the JSON representation.
Do not add or remove factual claims.
Do not add or remove citation numbers.
Do not wrap the JSON in Markdown code fences.
""".strip(),
)

QUESTION_SYSTEM_PROMPT = PromptTemplate(
    name="question-generation.system-json",
    version=1,
    text="""
Return only one valid JSON object with exactly one field named "questions".
"questions" must be a non-empty array of objects with exactly these fields:
- "text": the free-response study question
- "expected_answer": a grounded model answer
- "citation_numbers": a non-empty array of integers identifying only contexts
  that directly support the expected answer

Never invent source names, page numbers, excerpts, or context numbers.
Never return placeholder text.
Do not wrap the JSON object in Markdown code fences.
""".strip(),
)

QUESTION_JSON_REPAIR_PROMPT = PromptTemplate(
    name="question-generation.json-repair",
    version=1,
    text="""
Your previous response did not match the required question-bank JSON format.
Preserve every question, expected answer, and citation number from the previous
response.
Return the response again as one valid JSON object with exactly the field
"questions".
Each question must contain exactly "text", "expected_answer", and
"citation_numbers".
Correct only the JSON representation.
Do not add, remove, or rewrite factual content.
Do not add or remove citation numbers.
Do not wrap the JSON in Markdown code fences.
""".strip(),
)

ANSWER_EVALUATION_SYSTEM_PROMPT = PromptTemplate(
    name="answer-evaluation.system-json",
    version=1,
    text="""
Return only one valid JSON object with exactly these fields:
- "verdict": one of "incorrect", "partially_correct", or "correct"
- "score": a number from 0 to 1
- "feedback": a concise constructive explanation
- "missing_concepts": an array of strings

Judge only against the expected answer and supplied source contexts.
Do not invent facts, sources, citations, or review ratings.
Do not return "again", "hard", "good", or "easy".
Do not wrap the JSON object in Markdown code fences.
""".strip(),
)

ANSWER_EVALUATION_JSON_REPAIR_PROMPT = PromptTemplate(
    name="answer-evaluation.json-repair",
    version=1,
    text="""
Your previous response did not match the required answer-evaluation JSON format.
Preserve the verdict, score, feedback, and missing concepts from the previous
response.
Return one valid JSON object with exactly the fields "verdict", "score",
"feedback", and "missing_concepts".
Correct only the JSON representation.
Do not add, remove, or rewrite evaluation content.
Do not introduce review ratings, facts, sources, or citations.
Do not wrap the JSON object in Markdown code fences.
""".strip(),
)

_MAX_MODEL_RESPONSE_DIAGNOSTIC_CHARS = 4_000

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


def _add_model_failure_diagnostic(
    error: ValueError,
    *,
    phase: str,
    responses: tuple[tuple[str, str], ...],
) -> None:
    """Attach bounded model output to a terminal parsing error."""

    diagnostic_lines = [f"phase={phase}"]
    diagnostic_lines.extend(
        f"{name}={response[:_MAX_MODEL_RESPONSE_DIAGNOSTIC_CHARS]}" for name, response in responses
    )
    error.add_note("\n".join(diagnostic_lines))


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

    def generate(
        self,
        prompt: str,
        *,
        max_new_tokens: int | None = None,
    ) -> GenerationResult:
        """Generate and parse one grounded answer."""

        effective_max_new_tokens = self.max_new_tokens if max_new_tokens is None else max_new_tokens

        if effective_max_new_tokens < 1:
            raise ValueError("max_new_tokens must be positive")

        messages = [
            {
                "role": "system",
                "content": SYSTEM_PROMPT.text,
            },
            {
                "role": "user",
                "content": prompt,
            },
        ]
        raw_response = self._generate_raw_response(
            messages,
            max_new_tokens=effective_max_new_tokens,
        )

        try:
            result = parse_generation_response(raw_response)
            return replace(
                result,
                prompt_references=(SYSTEM_PROMPT.reference,),
            )
        except ValueError:
            # A small local model can produce grounded content but malformed JSON.
            # Give it exactly one format-repair attempt without permitting factual
            # or citation changes.
            repair_messages = [
                *messages,
                {
                    "role": "assistant",
                    "content": raw_response,
                },
                {
                    "role": "user",
                    "content": JSON_REPAIR_PROMPT.text,
                },
            ]
            repaired_response = self._generate_raw_response(
                repair_messages,
                max_new_tokens=effective_max_new_tokens,
            )
        result = parse_generation_response(repaired_response)

        return replace(
            result,
            prompt_references=(
                SYSTEM_PROMPT.reference,
                JSON_REPAIR_PROMPT.reference,
            ),
        )

    def generate_questions(
        self,
        prompt: str,
        *,
        max_new_tokens: int | None = None,
    ) -> QuestionGenerationResult:
        """Generate and parse one structured set of study questions."""

        effective_max_new_tokens = self.max_new_tokens if max_new_tokens is None else max_new_tokens
        if effective_max_new_tokens < 1:
            raise ValueError("max_new_tokens must be positive")

        messages = [
            {
                "role": "system",
                "content": QUESTION_SYSTEM_PROMPT.text,
            },
            {
                "role": "user",
                "content": prompt,
            },
        ]
        raw_response = self._generate_raw_response(
            messages,
            max_new_tokens=effective_max_new_tokens,
        )

        try:
            questions = parse_question_generation_response(
                raw_response,
            )
            return QuestionGenerationResult(
                questions=questions,
                prompt_references=(QUESTION_SYSTEM_PROMPT.reference,),
            )
        except ValueError:
            # Repair only the representation. Source grounding and question
            # contents must remain unchanged.
            repair_messages = [
                *messages,
                {
                    "role": "assistant",
                    "content": raw_response,
                },
                {
                    "role": "user",
                    "content": QUESTION_JSON_REPAIR_PROMPT.text,
                },
            ]
            repaired_response = self._generate_raw_response(
                repair_messages,
                max_new_tokens=effective_max_new_tokens,
            )

        try:
            questions = parse_question_generation_response(
                repaired_response,
            )
        except ValueError as error:
            _add_model_failure_diagnostic(
                error,
                phase="question-json-repair",
                responses=(
                    (
                        "initial_model_response",
                        raw_response,
                    ),
                    (
                        "repaired_model_response",
                        repaired_response,
                    ),
                ),
            )
            raise
        return QuestionGenerationResult(
            questions=questions,
            prompt_references=(
                QUESTION_SYSTEM_PROMPT.reference,
                QUESTION_JSON_REPAIR_PROMPT.reference,
            ),
        )

    def evaluate_answer(
        self,
        prompt: str,
        *,
        max_new_tokens: int | None = None,
    ) -> AnswerEvaluation:
        """Generate and parse one grounded learner-answer evaluation."""

        effective_max_new_tokens = self.max_new_tokens if max_new_tokens is None else max_new_tokens
        if effective_max_new_tokens < 1:
            raise ValueError("max_new_tokens must be positive")

        messages = [
            {
                "role": "system",
                "content": ANSWER_EVALUATION_SYSTEM_PROMPT.text,
            },
            {
                "role": "user",
                "content": prompt,
            },
        ]
        raw_response = self._generate_raw_response(
            messages,
            max_new_tokens=effective_max_new_tokens,
        )

        try:
            evaluation = parse_answer_evaluation(raw_response)
            return replace(
                evaluation,
                prompt_references=(ANSWER_EVALUATION_SYSTEM_PROMPT.reference,),
            )
        except ValueError:
            # Repair only the structured representation. The evaluator must
            # preserve its original judgment and may not change scheduling.
            repair_messages = [
                *messages,
                {
                    "role": "assistant",
                    "content": raw_response,
                },
                {
                    "role": "user",
                    "content": (ANSWER_EVALUATION_JSON_REPAIR_PROMPT.text),
                },
            ]
            repaired_response = self._generate_raw_response(
                repair_messages,
                max_new_tokens=effective_max_new_tokens,
            )

        evaluation = parse_answer_evaluation(repaired_response)
        return replace(
            evaluation,
            prompt_references=(
                ANSWER_EVALUATION_SYSTEM_PROMPT.reference,
                ANSWER_EVALUATION_JSON_REPAIR_PROMPT.reference,
            ),
        )

    def _generate_raw_response(
        self,
        messages: list[dict[str, str]],
        *,
        max_new_tokens: int,
    ) -> str:
        """Generate and extract one raw assistant response."""

        outputs = self._get_pipeline(max_new_tokens=max_new_tokens)(
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
        return self._extract_response(outputs)

    def _get_pipeline(
        self,
        *,
        max_new_tokens: int,
    ) -> ChatPipeline:
        """Load and configure the local model once."""

        if self._pipeline is None:
            self._pipeline = self._load_pipeline()

        # Transformers 5 expects generation settings in one place. Clearing
        # max_length avoids competing limits when max_new_tokens is configured.
        self._pipeline.generation_config.max_length = None
        self._pipeline.generation_config.max_new_tokens = max_new_tokens
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
