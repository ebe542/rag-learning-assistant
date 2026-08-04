"""Stable command-line entry point."""

from rag_learning_assistant.interfaces.cli.entrypoint import main

__all__ = ["main"]


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
