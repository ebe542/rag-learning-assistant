"""Command-specific CLI parser registration."""

from rag_learning_assistant.interfaces.cli.parsers.documents import (
    add_document_commands,
)
from rag_learning_assistant.interfaces.cli.parsers.packages import (
    add_package_commands,
)
from rag_learning_assistant.interfaces.cli.parsers.questions import (
    add_question_commands,
)
from rag_learning_assistant.interfaces.cli.parsers.retrieval import (
    add_retrieval_commands,
)
from rag_learning_assistant.interfaces.cli.parsers.reviews import (
    add_review_commands,
)
from rag_learning_assistant.interfaces.cli.parsers.summaries import (
    add_summary_commands,
)

__all__ = [
    "add_document_commands",
    "add_question_commands",
    "add_retrieval_commands",
    "add_review_commands",
    "add_summary_commands",
    "add_package_commands",
]
