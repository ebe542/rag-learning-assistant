import json
from pathlib import Path

import pytest

from rag_learning_assistant.interfaces.cli import commands, doctor, entrypoint
from rag_learning_assistant.interfaces.cli.doctor import (
    DependencyCheck,
    DoctorReport,
)
from rag_learning_assistant.interfaces.cli.parser import build_parser


def make_report(*, ready: bool) -> DoctorReport:
    return DoctorReport(
        python_version="3.13.14",
        python_supported=True,
        library_directory="personal-library",
        library_status="ready",
        dependencies=(
            DependencyCheck("PyMuPDF", "fitz", True),
            DependencyCheck("PyTorch", "torch", ready),
        ),
        cuda_available=True,
        gpu_name="Test GPU",
    )


def test_parser_accepts_doctor_options() -> None:
    args = build_parser().parse_args(["doctor", "--library", "personal-library", "--json"])

    assert args.command == "doctor"
    assert args.library == Path("personal-library")
    assert args.json_output is True


def test_entrypoint_dispatches_doctor(monkeypatch) -> None:
    calls: list[tuple[Path, bool]] = []
    monkeypatch.setattr(
        commands,
        "run_doctor",
        lambda library_directory, json_output: calls.append((library_directory, json_output)) or 0,
    )

    assert entrypoint.main(["doctor", "--library", "personal-library"]) == 0
    assert calls == [(Path("personal-library"), False)]


def test_run_doctor_prints_human_readable_ready_report(monkeypatch, capsys) -> None:
    monkeypatch.setattr(commands, "build_doctor_report", lambda directory: make_report(ready=True))

    assert commands.run_doctor(Path("personal-library")) == 0
    assert capsys.readouterr().out == (
        "RAG Learning Assistant diagnostics\n"
        "Python: 3.13.14 (supported)\n"
        "Library: personal-library (ready)\n"
        "Dependencies:\n"
        "- PyMuPDF: available\n"
        "- PyTorch: available\n"
        "GPU: Test GPU (CUDA available)\n"
        "Status: ready\n"
    )


def test_run_doctor_returns_failure_and_json_for_missing_dependency(
    monkeypatch,
    capsys,
) -> None:
    monkeypatch.setattr(commands, "build_doctor_report", lambda directory: make_report(ready=False))

    assert commands.run_doctor(Path("personal-library"), json_output=True) == 1
    payload = json.loads(capsys.readouterr().out)
    assert payload["ready"] is False
    assert payload["dependencies"][1] == {
        "name": "PyTorch",
        "module": "torch",
        "available": False,
    }


def test_build_doctor_report_inspects_library_and_dependencies(
    tmp_path: Path,
    monkeypatch,
) -> None:
    (tmp_path / "metadata.sqlite3").write_bytes(b"")
    (tmp_path / "vectors.faiss").write_bytes(b"")
    monkeypatch.setattr(doctor, "_module_available", lambda module: True)
    monkeypatch.setattr(doctor, "_probe_cuda", lambda torch_available: (True, "Test GPU"))

    report = doctor.build_doctor_report(tmp_path)

    assert report.library_status == "ready"
    assert all(dependency.available for dependency in report.dependencies)
    assert report.cuda_available is True
    assert report.gpu_name == "Test GPU"
    assert report.ready is True


@pytest.mark.parametrize(
    ("metadata", "vectors", "expected"),
    [
        (False, False, "not created"),
        (True, True, "ready"),
        (True, False, "incomplete"),
        (False, True, "incomplete"),
    ],
)
def test_library_status(
    tmp_path: Path,
    metadata: bool,
    vectors: bool,
    expected: str,
) -> None:
    if metadata:
        (tmp_path / "metadata.sqlite3").write_bytes(b"")
    if vectors:
        (tmp_path / "vectors.faiss").write_bytes(b"")

    assert doctor._library_status(tmp_path) == expected


def test_module_availability_handles_lookup_errors(monkeypatch) -> None:
    monkeypatch.setattr(
        doctor.importlib.util,
        "find_spec",
        lambda module: (_ for _ in ()).throw(ImportError("broken package")),
    )

    assert doctor._module_available("broken") is False


def test_cuda_probe_reports_unavailable_without_torch() -> None:
    assert doctor._probe_cuda(torch_available=False) == (False, None)


def test_cuda_probe_reports_available_gpu(monkeypatch) -> None:
    class FakeCuda:
        @staticmethod
        def is_available() -> bool:
            return True

        @staticmethod
        def get_device_name(index: int) -> str:
            assert index == 0
            return "Test GPU"

    class FakeTorch:
        cuda = FakeCuda()

    monkeypatch.setattr(doctor.importlib, "import_module", lambda module: FakeTorch())

    assert doctor._probe_cuda(torch_available=True) == (True, "Test GPU")
