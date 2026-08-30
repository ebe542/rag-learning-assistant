"""Serial execution of persisted package preparation requests."""

import threading
from collections.abc import Callable
from contextlib import suppress
from datetime import UTC, datetime, timedelta
from pathlib import Path
from uuid import UUID, uuid4

from rag_learning_assistant.application.learning_package import LearningPackageService
from rag_learning_assistant.application.package_preparation import PackagePreparationService
from rag_learning_assistant.learning import PackagePreparationStatus


class PackagePreparationWorker:
    """Process at most one leased request for one library."""

    def __init__(
        self,
        preparations: PackagePreparationService,
        package_service_factory: Callable[[Callable[[str], None]], LearningPackageService],
        upload_directory: Path,
        *,
        clock: Callable[[], datetime] = lambda: datetime.now(UTC),
        token_factory: Callable[[], UUID] = uuid4,
        lease_duration: timedelta = timedelta(minutes=5),
        heartbeat_interval: float = 60.0,
    ) -> None:
        self.preparations = preparations
        self.package_service_factory = package_service_factory
        self.upload_directory = upload_directory
        self.clock = clock
        self.token_factory = token_factory
        self.lease_duration = lease_duration
        self.heartbeat_interval = heartbeat_interval

    def run_once(self) -> bool:
        """Process one queued request and report whether work was claimed."""

        lease_token = self.token_factory()
        preparation = self.preparations.claim_next(
            lease_token=lease_token,
            now=self.clock(),
            lease_duration=self.lease_duration,
        )
        if preparation is None:
            return False

        current_status = preparation.status
        state_lock = threading.Lock()
        stop_heartbeat = threading.Event()
        heartbeat_errors: list[Exception] = []

        def heartbeat() -> None:
            while not stop_heartbeat.wait(self.heartbeat_interval):
                try:
                    with state_lock:
                        self.preparations.renew_lease(
                            preparation.id,
                            lease_token=lease_token,
                            now=self.clock(),
                            lease_duration=self.lease_duration,
                        )
                except Exception as error:
                    heartbeat_errors.append(error)
                    return

        heartbeat_thread = threading.Thread(
            target=heartbeat,
            name=f"package-preparation-heartbeat-{preparation.id}",
            daemon=True,
        )
        heartbeat_thread.start()

        def report_progress(phase: str) -> None:
            nonlocal current_status
            desired_status = {
                "index": PackagePreparationStatus.INDEXING,
                "summarize": PackagePreparationStatus.SUMMARIZING,
                "questions": PackagePreparationStatus.GENERATING_QUESTIONS,
            }.get(phase)
            if desired_status is None:
                return
            with state_lock:
                if heartbeat_errors:
                    raise RuntimeError(
                        "Package preparation lease renewal failed"
                    ) from heartbeat_errors[0]
                current_status = self._advance_to(
                    preparation.id,
                    lease_token=lease_token,
                    current_status=current_status,
                    desired_status=desired_status,
                )

        try:
            pdf_path = self.upload_directory / preparation.stored_filename
            if not pdf_path.is_file():
                raise FileNotFoundError(f"Stored package upload does not exist: {pdf_path}")
            package_service = self.package_service_factory(report_progress)
            package_service.prepare(
                name=preparation.name,
                pdf_path=pdf_path,
                question_count=preparation.question_count,
                preparation_id=preparation.id,
                source_filename=preparation.source_filename,
            )
            with state_lock:
                if heartbeat_errors:
                    raise RuntimeError(
                        "Package preparation lease renewal failed"
                    ) from heartbeat_errors[0]
                if current_status is not PackagePreparationStatus.GENERATING_QUESTIONS:
                    raise RuntimeError("Package preparation did not reach the final worker phase")
                self.preparations.complete(
                    preparation.id,
                    lease_token=lease_token,
                    now=self.clock(),
                )
            pdf_path.unlink(missing_ok=True)
        except Exception as error:
            with state_lock, suppress(ValueError):
                self.preparations.mark_failed(
                    preparation.id,
                    lease_token=lease_token,
                    now=self.clock(),
                    message=f"{type(error).__name__}: {error}"[:1000],
                )
        finally:
            stop_heartbeat.set()
            heartbeat_thread.join(timeout=max(self.heartbeat_interval * 2, 1.0))
        return True

    def _advance_to(
        self,
        preparation_id: UUID,
        *,
        lease_token: UUID,
        current_status: PackagePreparationStatus,
        desired_status: PackagePreparationStatus,
    ) -> PackagePreparationStatus:
        ordered = (
            PackagePreparationStatus.INDEXING,
            PackagePreparationStatus.SUMMARIZING,
            PackagePreparationStatus.GENERATING_QUESTIONS,
        )
        current_index = ordered.index(current_status)
        desired_index = ordered.index(desired_status)
        if desired_index < current_index:
            raise RuntimeError("Package pipeline attempted to move to an earlier phase")
        while current_index < desired_index:
            next_status = ordered[current_index + 1]
            self.preparations.advance(
                preparation_id,
                lease_token=lease_token,
                current_status=ordered[current_index],
                next_status=next_status,
                now=self.clock(),
                lease_duration=self.lease_duration,
            )
            current_index += 1
        return ordered[current_index]
