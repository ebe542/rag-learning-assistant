"""Register search and grounded question-answering commands."""

from pathlib import Path

from rag_learning_assistant.interfaces.cli.parsing import (
    DEFAULT_RESULT_LIMIT,
    SubcommandCollection,
    non_blank_text,
    positive_int,
)


def add_retrieval_commands(commands: SubcommandCollection) -> None:
    """Register commands backed by retrieval from a persistent index."""

    search_parser = commands.add_parser("search", help="Search an existing persistent index")
    search_parser.add_argument(
        "index_dir",
        type=Path,
        help="Directory containing the FAISS index and SQLite metadata",
    )
    search_parser.add_argument("query", type=non_blank_text, help="Question or search text")
    search_parser.add_argument(
        "--limit",
        type=positive_int,
        default=DEFAULT_RESULT_LIMIT,
        help=f"Maximum number of results (default: {DEFAULT_RESULT_LIMIT})",
    )

    ask_parser = commands.add_parser(
        "ask", help="Answer a question using an existing persistent index"
    )
    ask_parser.add_argument(
        "index_dir",
        type=Path,
        help="Directory containing the FAISS index and SQLite metadata",
    )
    ask_parser.add_argument(
        "question",
        type=non_blank_text,
        help="Question to answer from the indexed documents",
    )
    ask_parser.add_argument(
        "--limit",
        type=positive_int,
        default=DEFAULT_RESULT_LIMIT,
        help=f"Maximum number of source contexts (default: {DEFAULT_RESULT_LIMIT})",
    )
