from pathlib import Path

import pytest

from rag_learning_assistant import cli
from rag_learning_assistant.ingestion import TesseractPageOcr
from rag_learning_assistant.interfaces.cli import commands
from rag_learning_assistant.interfaces.cli.error_reporting import (
    default_log_path,
    write_diagnostic_log,
    write_exception_log,
)


def test_write_exception_log_preserves_technical_details(
    tmp_path: Path,
) -> None:
    log_path = tmp_path / "application.log"

    try:
        raise ValueError("Reduction must be supported by every section summary")
    except ValueError as error:
        error.add_note(
            "phase=question-json-repair\n"
            "initial_model_response=not JSON\n"
            "repaired_model_response=still not JSON"
        )
        result = write_exception_log(
            error,
            command="prepare",
            context={
                "library": "local-data/library",
            },
            log_path=log_path,
        )

    content = log_path.read_text(encoding="utf-8")

    assert result == log_path
    assert "command=prepare" in content
    assert "library=local-data/library" in content
    assert "ValueError" in content
    assert "Reduction must be supported by every section summary" in content
    assert "Traceback (most recent call last)" in content
    assert "phase=question-json-repair" in content
    assert "initial_model_response=not JSON" in content
    assert "repaired_model_response=still not JSON" in content


def test_default_log_path_honors_environment_override(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setenv(
        "RAG_LEARN_LOG_DIR",
        str(tmp_path),
    )

    assert default_log_path() == (tmp_path / "application.log")


def test_console_entrypoint_logs_exception_without_traceback(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    log_path = tmp_path / "application.log"
    logged: list[tuple[Exception, str]] = []

    def fail_dispatch(argv) -> int:
        raise ValueError("Reduction must be supported by every section summary")

    def fake_write_exception_log(
        error: Exception,
        *,
        command: str,
        context: dict[str, object],
        log_path: Path | None = None,
    ) -> Path:
        logged.append((error, command))
        return tmp_path / "application.log"

    monkeypatch.setattr(
        cli.entrypoint,
        "main",
        fail_dispatch,
    )
    monkeypatch.setattr(
        cli,
        "write_exception_log",
        fake_write_exception_log,
    )

    exit_code = cli.main(
        [
            "prepare",
            "document.pdf",
            "--library",
            "library",
        ]
    )

    error_output = capsys.readouterr().err

    assert exit_code == 1
    assert len(logged) == 1
    assert logged[0][1] == "prepare"
    assert "Command failed:" in error_output
    assert "Reduction must be supported by every section summary" in error_output
    assert str(log_path) in error_output
    assert "Traceback" not in error_output


def test_console_entrypoint_hides_logging_failure(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    def fail_dispatch(argv) -> int:
        raise RuntimeError("Model generation failed")

    def fail_logging(*args, **kwargs) -> Path:
        raise PermissionError("Log directory is not writable")

    monkeypatch.setattr(
        cli.entrypoint,
        "main",
        fail_dispatch,
    )
    monkeypatch.setattr(
        cli,
        "write_exception_log",
        fail_logging,
    )

    exit_code = cli.main(["prepare"])

    error_output = capsys.readouterr().err

    assert exit_code == 1
    assert "Command failed: Model generation failed" in error_output
    assert "Technical details could not be written" in error_output
    assert "Log directory is not writable" not in error_output
    assert "Traceback" not in error_output


def test_console_entrypoint_preserves_parser_exit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_dispatch(argv) -> int:
        raise SystemExit(2)

    def fail_if_logged(*args, **kwargs) -> Path:
        pytest.fail("Parser exits must not be written as internal errors")

    monkeypatch.setattr(
        cli.entrypoint,
        "main",
        fail_dispatch,
    )
    monkeypatch.setattr(
        cli,
        "write_exception_log",
        fail_if_logged,
    )

    with pytest.raises(SystemExit) as exc_info:
        cli.main(["unknown-command"])

    assert exc_info.value.code == 2


def test_write_diagnostic_log_records_non_fatal_warning(
    tmp_path: Path,
) -> None:
    log_path = tmp_path / "application.log"

    result = write_diagnostic_log(
        "cmsOpenProfileFromMem failed",
        source="pymupdf",
        context={
            "document": "Dummy.pdf",
        },
        log_path=log_path,
    )

    content = log_path.read_text(encoding="utf-8")

    assert result == log_path
    assert "WARNING" in content
    assert "source=pymupdf" in content
    assert "document=Dummy.pdf" in content
    assert "cmsOpenProfileFromMem failed" in content
    assert "Traceback" not in content


def test_pdf_extractor_builder_logs_mupdf_diagnostics(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, str, dict[str, object]]] = []

    def fake_write_diagnostic_log(
        message: str,
        *,
        source: str,
        context: dict[str, object],
        log_path: Path | None = None,
    ) -> Path:
        calls.append(
            (
                message,
                source,
                context,
            )
        )
        return Path("application.log")

    monkeypatch.setattr(
        commands,
        "write_diagnostic_log",
        fake_write_diagnostic_log,
        raising=False,
    )

    extractor = commands.build_pdf_extractor()
    handler = extractor.diagnostic_handler

    assert handler is not None

    document_path = Path("local-data/books/Dummy.pdf")

    handler(
        document_path,
        "cmsOpenProfileFromMem failed",
    )

    assert calls == [
        (
            "cmsOpenProfileFromMem failed",
            "pymupdf",
            {
                "document": str(document_path),
            },
        )
    ]


def test_pdf_extractor_builder_configures_ocr_languages(monkeypatch) -> None:
    monkeypatch.setenv("RAG_LEARN_OCR_LANGUAGES", "eng")

    extractor = commands.build_pdf_extractor()

    assert isinstance(extractor.ocr, TesseractPageOcr)
    assert extractor.ocr.languages == "eng"
