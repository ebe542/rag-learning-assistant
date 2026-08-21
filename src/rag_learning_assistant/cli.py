"""Stable command-line entry point with user-facing error reporting."""

import sys
from collections.abc import Sequence

from rag_learning_assistant.interfaces.cli import entrypoint
from rag_learning_assistant.interfaces.cli.error_reporting import (
    write_exception_log,
)


def main(
    argv: Sequence[str] | None = None,
) -> int:
    """Run the CLI without exposing technical tracebacks to users."""

    arguments = list(argv) if argv is not None else sys.argv[1:]
    command = arguments[0] if arguments else "unknown"

    try:
        return entrypoint.main(argv)
    except Exception as error:
        # Expected parser exits and Ctrl+C derive from BaseException rather
        # than Exception and therefore keep their normal terminal behavior.
        try:
            log_path = write_exception_log(
                error,
                command=command,
                context={
                    "argument_count": max(
                        0,
                        len(arguments) - 1,
                    ),
                },
            )
        except Exception:
            # Error reporting must never replace the original application
            # failure with another traceback.
            print(
                f"Command failed: {error}",
                file=sys.stderr,
            )
            print(
                "Technical details could not be written to the application log.",
                file=sys.stderr,
            )
            return 1

        print(
            f"Command failed: {error}",
            file=sys.stderr,
        )
        print(
            f"Technical details: {log_path}",
            file=sys.stderr,
        )
        return 1


__all__ = ["main"]


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
