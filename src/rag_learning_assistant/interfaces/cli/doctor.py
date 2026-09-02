"""Local environment diagnostics for the product CLI."""

import importlib
import importlib.util
import os
import sys
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path

from rag_learning_assistant.ingestion import DEFAULT_OCR_LANGUAGES


@dataclass(frozen=True, slots=True)
class DependencyCheck:
    """Describe whether one runtime capability can be imported."""

    name: str
    module: str
    available: bool


@dataclass(frozen=True, slots=True)
class OcrCheck:
    """Describe optional Tesseract language-data readiness."""

    available: bool
    tessdata_directory: str | None
    languages: tuple[str, ...]
    missing_languages: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class DoctorReport:
    """Describe local readiness without changing application state."""

    python_version: str
    python_supported: bool
    library_directory: str
    library_status: str
    dependencies: tuple[DependencyCheck, ...]
    ocr: OcrCheck
    cuda_available: bool
    gpu_name: str | None

    @property
    def ready(self) -> bool:
        """Return whether the complete local product workflow can run."""

        return (
            self.python_supported
            and self.library_status != "incomplete"
            and all(check.available for check in self.dependencies)
        )

    def as_json(self) -> dict[str, object]:
        """Return a stable machine-readable representation."""

        payload = asdict(self)
        payload["ready"] = self.ready
        return payload


def build_doctor_report(library_directory: Path) -> DoctorReport:
    """Inspect the current process and selected library without writing files."""

    dependencies = tuple(
        DependencyCheck(name=name, module=module, available=_module_available(module))
        for name, module in (
            ("PyMuPDF", "fitz"),
            ("Sentence Transformers", "sentence_transformers"),
            ("Transformers", "transformers"),
            ("PyTorch", "torch"),
            ("FAISS", "faiss"),
        )
    )
    cuda_available, gpu_name = _probe_cuda(
        torch_available=next(check.available for check in dependencies if check.module == "torch")
    )

    version = sys.version_info
    return DoctorReport(
        python_version=f"{version.major}.{version.minor}.{version.micro}",
        python_supported=(3, 11) <= version[:2] < (3, 14),
        library_directory=str(library_directory),
        library_status=_library_status(library_directory),
        dependencies=dependencies,
        ocr=_ocr_check(os.environ),
        cuda_available=cuda_available,
        gpu_name=gpu_name,
    )


def _module_available(module: str) -> bool:
    """Check import availability without importing a potentially heavy package."""

    try:
        return importlib.util.find_spec(module) is not None
    except (ImportError, ValueError):
        return False


def _library_status(library_directory: Path) -> str:
    metadata_exists = (library_directory / "metadata.sqlite3").is_file()
    vectors_exist = (library_directory / "vectors.faiss").is_file()
    if metadata_exists and vectors_exist:
        return "ready"
    if not metadata_exists and not vectors_exist:
        return "not created"
    return "incomplete"


def _ocr_check(environment: Mapping[str, str]) -> OcrCheck:
    """Inspect optional Tesseract data without invoking OCR."""

    configured_languages = environment.get(
        "RAG_LEARN_OCR_LANGUAGES",
        DEFAULT_OCR_LANGUAGES,
    )
    languages = tuple(
        language
        for language in (part.strip() for part in configured_languages.split("+"))
        if language
    )
    if not languages:
        languages = tuple(DEFAULT_OCR_LANGUAGES.split("+"))

    configured_directory = environment.get("TESSDATA_PREFIX", "").strip()
    tessdata_directory = Path(configured_directory) if configured_directory else None
    if tessdata_directory is None or not tessdata_directory.is_dir():
        missing_languages = languages
    else:
        missing_languages = tuple(
            language
            for language in languages
            if not (tessdata_directory / f"{language}.traineddata").is_file()
        )

    return OcrCheck(
        available=tessdata_directory is not None
        and tessdata_directory.is_dir()
        and not missing_languages,
        tessdata_directory=str(tessdata_directory) if tessdata_directory is not None else None,
        languages=languages,
        missing_languages=missing_languages,
    )


def _probe_cuda(*, torch_available: bool) -> tuple[bool, str | None]:
    """Inspect CUDA only when PyTorch is installed."""

    if not torch_available:
        return False, None

    try:
        torch = importlib.import_module("torch")
        if not torch.cuda.is_available():
            return False, None
        return True, str(torch.cuda.get_device_name(0))
    except Exception:
        return False, None
