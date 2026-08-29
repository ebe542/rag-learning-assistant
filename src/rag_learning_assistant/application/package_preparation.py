"""Store validated package uploads without starting model work."""

import shutil
from collections.abc import Callable
from pathlib import Path
from typing import BinaryIO, Protocol
from uuid import UUID, uuid4

from rag_learning_assistant.learning.preparations import PackagePreparation


class PackagePreparationRepository(Protocol):
    """Persist and discover pending package preparation requests."""

    def save(self, preparation: PackagePreparation) -> None: ...

    def find_by_name(self, name: str) -> PackagePreparation | None: ...

    def list_all(self) -> list[PackagePreparation]: ...


class PackagePreparationService:
    """Own validated uploads under UUID-derived internal filenames."""

    def __init__(
        self,
        repository: PackagePreparationRepository,
        upload_directory: Path,
        *,
        id_factory: Callable[[], UUID] = uuid4,
    ) -> None:
        self.repository = repository
        self.upload_directory = upload_directory
        self.id_factory = id_factory

    def list_all(self) -> list[PackagePreparation]:
        return self.repository.list_all()

    def store(
        self,
        *,
        name: str,
        source_filename: str,
        question_count: int,
        size_bytes: int,
        source: BinaryIO,
    ) -> PackagePreparation:
        """Atomically store one PDF and then register its pending request."""

        if self.repository.find_by_name(name) is not None:
            raise ValueError(f"Learning package already exists: {name}")

        preparation_id = self.id_factory()
        stored_filename = f"{preparation_id}.pdf"
        preparation = PackagePreparation(
            id=preparation_id,
            name=name,
            source_filename=_display_filename(source_filename),
            stored_filename=stored_filename,
            question_count=question_count,
            size_bytes=size_bytes,
        )
        self.upload_directory.mkdir(parents=True, exist_ok=True)
        target = self.upload_directory / stored_filename
        temporary = self.upload_directory / f"{stored_filename}.part"
        if target.exists() or temporary.exists():
            raise ValueError("Package preparation ID already exists")

        try:
            with temporary.open("xb") as destination:
                shutil.copyfileobj(source, destination, length=64 * 1024)
            temporary.replace(target)
            self.repository.save(preparation)
        except Exception:
            temporary.unlink(missing_ok=True)
            target.unlink(missing_ok=True)
            raise
        return preparation


def _display_filename(filename: str) -> str:
    """Keep only a display-safe basename; storage never uses user input."""

    return filename.replace("\\", "/").rsplit("/", 1)[-1]
