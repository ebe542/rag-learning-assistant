"""Answer generation with explicit source grounding."""

from rag_learning_assistant.generation.generator import TextGenerator
from rag_learning_assistant.generation.huggingface import (
    HuggingFaceTextGenerator,
)
from rag_learning_assistant.generation.identity import (
    GenerationIdentity,
)
from rag_learning_assistant.generation.models import (
    Citation,
    GenerationResult,
    GroundedAnswer,
)
from rag_learning_assistant.generation.prompts import (
    PromptReference,
    PromptTemplate,
)

__all__ = [
    "Citation",
    "GenerationIdentity",
    "GenerationResult",
    "GroundedAnswer",
    "HuggingFaceTextGenerator",
    "PromptReference",
    "PromptTemplate",
    "TextGenerator",
]
