"""Persist technical CLI failures without exposing tracebacks to users."""

import logging
import os
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path

_MAX_LOG_BYTES = 1_000_000
_BACKUP_COUNT = 3


def default_log_path() -> Path:
    """Return the central per-user application log path."""

    configured_directory = os.environ.get("RAG_LEARN_LOG_DIR")
    if configured_directory:
        return Path(configured_directory) / "application.log"

    if os.name == "nt":
        local_app_data = os.environ.get("LOCALAPPDATA")
        root = Path(local_app_data) if local_app_data else Path.home() / "AppData" / "Local"
        return root / "rag-learning-assistant" / "logs" / "application.log"

    if sys.platform == "darwin":
        return Path.home() / "Library" / "Logs" / "rag-learning-assistant" / "application.log"

    state_home = os.environ.get("XDG_STATE_HOME")
    root = Path(state_home) if state_home else Path.home() / ".local" / "state"
    return root / "rag-learning-assistant" / "logs" / "application.log"


def write_exception_log(
    error: Exception,
    *,
    command: str,
    context: dict[str, object],
    log_path: Path | None = None,
) -> Path:
    """Write one failure with traceback to the rotating user log."""

    target = log_path if log_path is not None else default_log_path()
    target.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Use a private logger and close its handler immediately so Windows never
    # retains a file lock after the command exits or a test finishes.
    logger = logging.Logger(
        "rag-learning-assistant.error",
        level=logging.ERROR,
    )
    handler = RotatingFileHandler(
        target,
        maxBytes=_MAX_LOG_BYTES,
        backupCount=_BACKUP_COUNT,
        encoding="utf-8",
    )
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logger.addHandler(handler)

    context_text = " ".join(f"{key}={value}" for key, value in sorted(context.items()))

    try:
        logger.error(
            "command=%s %s",
            command,
            context_text,
            exc_info=(
                type(error),
                error,
                error.__traceback__,
            ),
        )
    finally:
        handler.close()
        logger.removeHandler(handler)

    return target


def write_diagnostic_log(
    message: str,
    *,
    source: str,
    context: dict[str, object],
    log_path: Path | None = None,
) -> Path:
    """Write one non-fatal diagnostic to the rotating user log."""

    target = log_path if log_path is not None else default_log_path()
    target.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    logger = logging.Logger(
        "rag-learning-assistant.diagnostic",
        level=logging.WARNING,
    )
    handler = RotatingFileHandler(
        target,
        maxBytes=_MAX_LOG_BYTES,
        backupCount=_BACKUP_COUNT,
        encoding="utf-8",
    )
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logger.addHandler(handler)

    context_text = " ".join(f"{key}={value}" for key, value in sorted(context.items()))

    try:
        logger.warning(
            "source=%s %s message=%s",
            source,
            context_text,
            message,
        )
    finally:
        handler.close()
        logger.removeHandler(handler)

    return target
