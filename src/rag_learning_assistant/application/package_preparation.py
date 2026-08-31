"""Store validated package uploads without starting model work."""

import shutil
from collections.abc import Callable
from datetime import datetime, timedelta
from hashlib import sha256
from pathlib import Path
from typing import BinaryIO, Protocol
from uuid import UUID, uuid4

from rag_learning_assistant.learning.languages import LearningLanguage
from rag_learning_assistant.learning.preparations import (
    PackagePreparation,
    PackagePreparationStatus,
)


class PackagePreparationRepository(Protocol):
    """Persist and discover pending package preparation requests."""

    def save(self, preparation: PackagePreparation) -> None: ...

    def find_by_name(self, name: str) -> PackagePreparation | None: ...

    def find_by_content_hash(self, content_sha256: str) -> PackagePreparation | None: ...

    def update_content_hash(self, preparation_id: UUID, content_sha256: str) -> None: ...

    def list_all(self) -> list[PackagePreparation]: ...

    def claim_next(
        self,
        *,
        lease_token: UUID,
        now: datetime,
        lease_duration: timedelta,
    ) -> PackagePreparation | None: ...

    def advance(
        self,
        preparation_id: UUID,
        *,
        lease_token: UUID,
        current_status: PackagePreparationStatus,
        next_status: PackagePreparationStatus,
        now: datetime,
        lease_duration: timedelta,
    ) -> PackagePreparation: ...

    def mark_failed(
        self,
        preparation_id: UUID,
        *,
        lease_token: UUID,
        now: datetime,
        message: str,
    ) -> PackagePreparation: ...

    def retry_failed(self, preparation_id: UUID, *, now: datetime) -> PackagePreparation: ...

    def delete_failed(self, preparation_id: UUID) -> PackagePreparation: ...

    def renew_lease(
        self,
        preparation_id: UUID,
        *,
        lease_token: UUID,
        now: datetime,
        lease_duration: timedelta,
    ) -> PackagePreparation: ...

    def complete(
        self,
        preparation_id: UUID,
        *,
        lease_token: UUID,
        now: datetime,
    ) -> PackagePreparation: ...


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

    def backfill_missing_hashes(self) -> None:
        """Migrate uploads stored before duplicate detection was introduced."""

        for preparation in self.repository.list_all():
            if preparation.content_sha256 is not None:
                continue
            path = self.upload_directory / preparation.stored_filename
            if not path.is_file():
                continue
            digest = sha256()
            with path.open("rb") as source:
                while chunk := source.read(64 * 1024):
                    digest.update(chunk)
            self.repository.update_content_hash(preparation.id, digest.hexdigest())

    def claim_next(
        self,
        *,
        lease_token: UUID,
        now: datetime,
        lease_duration: timedelta,
    ) -> PackagePreparation | None:
        return self.repository.claim_next(
            lease_token=lease_token,
            now=now,
            lease_duration=lease_duration,
        )

    def advance(
        self,
        preparation_id: UUID,
        *,
        lease_token: UUID,
        current_status: PackagePreparationStatus,
        next_status: PackagePreparationStatus,
        now: datetime,
        lease_duration: timedelta,
    ) -> PackagePreparation:
        return self.repository.advance(
            preparation_id,
            lease_token=lease_token,
            current_status=current_status,
            next_status=next_status,
            now=now,
            lease_duration=lease_duration,
        )

    def mark_failed(
        self,
        preparation_id: UUID,
        *,
        lease_token: UUID,
        now: datetime,
        message: str,
    ) -> PackagePreparation:
        return self.repository.mark_failed(
            preparation_id,
            lease_token=lease_token,
            now=now,
            message=message,
        )

    def renew_lease(
        self,
        preparation_id: UUID,
        *,
        lease_token: UUID,
        now: datetime,
        lease_duration: timedelta,
    ) -> PackagePreparation:
        return self.repository.renew_lease(
            preparation_id,
            lease_token=lease_token,
            now=now,
            lease_duration=lease_duration,
        )

    def complete(
        self,
        preparation_id: UUID,
        *,
        lease_token: UUID,
        now: datetime,
    ) -> PackagePreparation:
        return self.repository.complete(
            preparation_id,
            lease_token=lease_token,
            now=now,
        )

    def retry_failed(self, preparation_id: UUID, *, now: datetime) -> PackagePreparation:
        return self.repository.retry_failed(preparation_id, now=now)

    def delete_failed(self, preparation_id: UUID) -> PackagePreparation:
        preparation = self.repository.delete_failed(preparation_id)
        (self.upload_directory / preparation.stored_filename).unlink(missing_ok=True)
        return preparation

    def store(
        self,
        *,
        name: str,
        source_filename: str,
        question_count: int,
        size_bytes: int,
        content_sha256: str,
        source: BinaryIO,
        learning_language: LearningLanguage = LearningLanguage.SAME_AS_DOCUMENT,
    ) -> PackagePreparation:
        """Atomically store one PDF and then register its pending request."""

        if self.repository.find_by_name(name) is not None:
            raise ValueError(f"Learning package already exists: {name}")
        duplicate = self.repository.find_by_content_hash(content_sha256)
        if duplicate is not None:
            raise ValueError(f"This PDF is already queued as {duplicate.name}")

        preparation_id = self.id_factory()
        stored_filename = f"{preparation_id}.pdf"
        preparation = PackagePreparation(
            id=preparation_id,
            name=name,
            source_filename=_display_filename(source_filename),
            stored_filename=stored_filename,
            question_count=question_count,
            size_bytes=size_bytes,
            content_sha256=content_sha256,
            learning_language=learning_language,
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
