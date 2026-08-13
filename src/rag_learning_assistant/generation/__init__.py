"""Answer generation with explicit source grounding."""

from rag_learning_assistant.generation.cache import (
    CachedSummaryBatch,
    SummaryBatchCache,
)
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
from rag_learning_assistant.generation.sqlite_cache import SqliteSummaryCache
from rag_learning_assistant.generation.summary_repository import (
    DocumentSummaryRepository,
    PersistedDocumentSummary,
    SqliteDocumentSummaryRepository,
)

__all__ = [
    "CachedSummaryBatch",
    "Citation",
    "DocumentSummaryRepository",
    "GenerationIdentity",
    "GenerationResult",
    "GroundedAnswer",
    "HuggingFaceTextGenerator",
    "PersistedDocumentSummary",
    "PromptReference",
    "PromptTemplate",
    "SqliteDocumentSummaryRepository",
    "SqliteSummaryCache",
    "SummaryBatchCache",
    "TextGenerator",
]
