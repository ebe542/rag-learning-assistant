"""Local environment diagnostics for the product CLI."""

import importlib
import importlib.util
import sys
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class DependencyCheck:
    """Describe whether one runtime capability can be imported."""

    name: str
    module: str
    available: bool


@dataclass(frozen=True, slots=True)
class DoctorReport:
    """Describe local readiness without changing application state."""

    python_version: str
    python_supported: bool
    library_directory: str
    library_status: str
    dependencies: tuple[DependencyCheck, ...]
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
