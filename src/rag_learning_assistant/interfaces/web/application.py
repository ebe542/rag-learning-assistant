"""Compose the local FastAPI application."""

from dataclasses import dataclass
from datetime import UTC, datetime, tzinfo
from hashlib import sha256
from pathlib import Path
from typing import Annotated, BinaryIO, Protocol
from urllib.parse import urlencode
from uuid import UUID

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse, RedirectResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.middleware.trustedhost import TrustedHostMiddleware

from rag_learning_assistant.application import (
    LearningPackageNotFoundError,
    LearningProgressReport,
)
from rag_learning_assistant.application.review import DueQuestion
from rag_learning_assistant.generation import PersistedDocumentSummary
from rag_learning_assistant.interfaces.web.libraries import LibraryListItem
from rag_learning_assistant.learning import (
    LearningLanguage,
    LearningPackage,
    PackagePreparation,
    QuestionBank,
    StudyAttempt,
)
from rag_learning_assistant.library import DocumentLanguage

WEB_ROOT = Path(__file__).resolve().parent
MAX_PDF_UPLOAD_BYTES = 25 * 1024 * 1024


class PackageCatalog(Protocol):
    """Supply learning packages without coupling routes to SQLite."""

    def list_packages(self) -> list[LearningPackage]: ...

    def get_package(self, name: str) -> LearningPackage: ...


class SummaryCatalog(Protocol):
    """Supply one exact persisted summary selected by a package."""

    def get_document_summary(
        self,
        document_id: UUID,
        identity_fingerprint: str,
    ) -> PersistedDocumentSummary: ...


class QuestionCatalog(Protocol):
    """Supply one exact persisted question bank selected by a package."""

    def get_document_bank(
        self,
        document_id: UUID,
        identity_fingerprint: str,
    ) -> QuestionBank: ...


class PackageStudy(Protocol):
    """Select and record study questions through a package name."""

    def next_due(self, package_name: str, *, as_of: datetime) -> DueQuestion | None: ...

    def record_answer(
        self,
        package_name: str,
        question_number: int,
        *,
        answer_text: str,
        answered_at: datetime,
    ) -> StudyAttempt: ...


class ProgressReporting(Protocol):
    """Build a current read-only progress report for one package."""

    def report(self, package_name: str, *, as_of: datetime) -> LearningProgressReport: ...


class LibraryManagement(Protocol):
    """Create libraries and open one within the configured local workspace."""

    @property
    def current_directory(self) -> Path | None: ...

    @property
    def current_library(self) -> LibraryListItem | None: ...

    def list_libraries(self) -> tuple[LibraryListItem, ...]: ...

    def create_library(self, name: str) -> LibraryListItem: ...

    def select_library(self, library_id: UUID) -> LibraryListItem: ...

    def rename_library(self, library_id: UUID, name: str) -> LibraryListItem: ...

    def delete_library(
        self,
        library_id: UUID,
        *,
        confirmation: str,
        delete_contents: bool,
    ) -> None: ...

    def list_package_preparations(self) -> list[PackagePreparation]: ...

    def get_document_language(self, document_id: UUID) -> DocumentLanguage: ...

    def store_package_upload(
        self,
        *,
        name: str,
        source_filename: str,
        question_count: int,
        size_bytes: int,
        content_sha256: str,
        learning_language: LearningLanguage,
        source: BinaryIO,
    ) -> PackagePreparation: ...

    def retry_package_preparation(self, preparation_id: UUID) -> PackagePreparation: ...

    def delete_failed_package_preparation(
        self,
        preparation_id: UUID,
    ) -> PackagePreparation: ...

    def rename_package(self, name: str, new_name: str) -> LearningPackage: ...

    def delete_package(self, name: str, *, confirmation: str) -> None: ...


@dataclass(frozen=True, slots=True)
class PackageListItem:
    """Contain only the learning-package fields rendered by the start page."""

    name: str
    status: str
    status_label: str
    has_detail: bool
    preparation_id: UUID | None
    failure_message: str | None
    document_language_label: str
    learning_language_label: str


def _document_language_label(language: DocumentLanguage) -> str:
    return {
        DocumentLanguage.GERMAN: "German",
        DocumentLanguage.ENGLISH: "English",
        DocumentLanguage.UNKNOWN: "Unknown",
    }[language]


def _learning_language_label(
    language: LearningLanguage,
    document_language: DocumentLanguage | None = None,
) -> str:
    if language is LearningLanguage.SAME_AS_DOCUMENT:
        if document_language is None:
            return "Same as document"
        return _document_language_label(language.resolve(document_language))
    return "German" if language is LearningLanguage.GERMAN else "English"


def format_local_datetime(
    value: datetime,
    *,
    timezone: tzinfo | None = None,
) -> str:
    """Format an aware timestamp for the local user-facing interface."""

    local_value = value.astimezone(timezone)
    return f"{local_value:%d.%m.%Y, %H:%M} (local time)"


def create_app(
    packages: PackageCatalog,
    summaries: SummaryCatalog,
    questions: QuestionCatalog,
    study: PackageStudy,
    progress: ProgressReporting,
    *,
    libraries: LibraryManagement,
) -> FastAPI:
    """Create an isolated web application suitable for runtime and tests."""

    app = FastAPI(
        title="RAG Learning Assistant",
        docs_url=None,
        redoc_url=None,
        openapi_url=None,
    )
    app.add_middleware(
        TrustedHostMiddleware,
        allowed_hosts=["127.0.0.1", "localhost", "testserver"],
    )
    app.mount(
        "/static",
        StaticFiles(directory=WEB_ROOT / "static"),
        name="static",
    )

    def navigation_context(_: Request) -> dict[str, LibraryListItem | None]:
        return {"current_library": libraries.current_library}

    templates = Jinja2Templates(
        directory=WEB_ROOT / "templates",
        context_processors=[navigation_context],
    )

    def package_items() -> tuple[PackageListItem, ...]:
        stored_packages = packages.list_packages()
        stored_packages_by_name = {package.name.casefold(): package for package in stored_packages}
        prepared = []
        for package in stored_packages:
            document_language = libraries.get_document_language(package.document_id)
            prepared.append(
                PackageListItem(
                    name=package.name,
                    status=package.status.value,
                    status_label=package.status.value.capitalize(),
                    has_detail=True,
                    preparation_id=None,
                    failure_message=None,
                    document_language_label=_document_language_label(document_language),
                    learning_language_label=_learning_language_label(
                        package.learning_language,
                        document_language,
                    ),
                )
            )
        items = {item.name.casefold(): item for item in prepared}
        for preparation in libraries.list_package_preparations():
            stored_package = stored_packages_by_name.get(preparation.name.casefold())
            document_language = (
                libraries.get_document_language(stored_package.document_id)
                if stored_package is not None
                else None
            )
            item = PackageListItem(
                name=preparation.name,
                status=preparation.status.value,
                status_label=preparation.status.value.replace("_", " ").capitalize(),
                has_detail=False,
                preparation_id=preparation.id,
                failure_message=_friendly_preparation_error(preparation.failure_message),
                document_language_label=(
                    _document_language_label(document_language)
                    if document_language is not None
                    else "Not detected yet"
                ),
                learning_language_label=_learning_language_label(
                    preparation.learning_language,
                    document_language,
                ),
            )
            items[item.name.casefold()] = item
        return tuple(sorted(items.values(), key=lambda item: item.name.casefold()))

    def package_list_context() -> dict[str, object]:
        items = package_items()
        return {
            "packages": items,
            "refresh_packages": any(
                item.status in {"pending", "indexing", "summarizing", "generating_questions"}
                for item in items
            ),
        }

    def render_package_create(
        request: Request,
        *,
        error_message: str | None = None,
        status_code: int = 200,
    ) -> HTMLResponse:
        return templates.TemplateResponse(
            request=request,
            name="package_create.html",
            context={"error_message": error_message},
            status_code=status_code,
        )

    def render_library_management(
        request: Request,
        *,
        error_message: str | None = None,
        selected_library_id: UUID | None = None,
        status_code: int = 200,
    ) -> HTMLResponse:
        library_items = libraries.list_libraries()
        return templates.TemplateResponse(
            request=request,
            name="library_management.html",
            context={
                "error_message": error_message,
                "libraries": library_items,
                "selected_library": next(
                    (item for item in library_items if item.id == selected_library_id),
                    None,
                ),
            },
            status_code=status_code,
        )

    @app.get("/", response_class=HTMLResponse)
    def home(request: Request) -> HTMLResponse:
        return templates.TemplateResponse(
            request=request,
            name="home.html",
            context={"libraries": libraries.list_libraries()},
        )

    @app.get("/library", response_class=HTMLResponse)
    def library_detail(request: Request) -> Response:
        current_library = libraries.current_library
        if current_library is None:
            return RedirectResponse(request.url_for("home"), status_code=303)
        return templates.TemplateResponse(
            request=request,
            name="library_detail.html",
            context={
                "library_name": current_library.name,
                **package_list_context(),
            },
        )

    @app.get("/library/packages/status", response_class=HTMLResponse)
    def library_package_status(request: Request) -> Response:
        if libraries.current_library is None:
            raise HTTPException(status_code=409, detail="No library is open")
        return templates.TemplateResponse(
            request=request,
            name="_package_list.html",
            context=package_list_context(),
        )

    @app.get("/libraries/manage", response_class=HTMLResponse)
    def library_management(
        request: Request,
        library_id: UUID | None = None,
    ) -> HTMLResponse:
        return render_library_management(request, selected_library_id=library_id)

    @app.post("/libraries", response_class=HTMLResponse)
    def create_library(request: Request, name: str = Form()) -> Response:
        _require_same_origin(request)
        try:
            libraries.create_library(name)
        except ValueError as error:
            return render_library_management(
                request,
                error_message=str(error),
                status_code=422,
            )
        return RedirectResponse(request.url_for("library_management"), status_code=303)

    @app.post("/libraries/rename", response_class=HTMLResponse)
    def rename_library(
        request: Request,
        library_id: Annotated[UUID, Form()],
        name: Annotated[str, Form()],
    ) -> Response:
        _require_same_origin(request)
        try:
            renamed = libraries.rename_library(library_id, name)
        except (LookupError, ValueError) as error:
            return render_library_management(
                request,
                error_message=str(error),
                selected_library_id=library_id,
                status_code=422,
            )
        location = f"{request.url_for('library_management')}?library_id={renamed.id}"
        return RedirectResponse(location, status_code=303)

    @app.post("/libraries/delete", response_class=HTMLResponse)
    def delete_library(
        request: Request,
        library_id: Annotated[UUID, Form()],
        confirmation: Annotated[str, Form()],
        delete_contents: Annotated[str | None, Form()] = None,
    ) -> Response:
        _require_same_origin(request)
        try:
            libraries.delete_library(
                library_id,
                confirmation=confirmation,
                delete_contents=delete_contents == "yes",
            )
        except (LookupError, ValueError) as error:
            return render_library_management(
                request,
                error_message=str(error),
                selected_library_id=library_id,
                status_code=422,
            )
        return RedirectResponse(request.url_for("library_management"), status_code=303)

    @app.post("/libraries/select", response_class=HTMLResponse)
    def select_library(
        request: Request,
        library_id: Annotated[UUID, Form()],
    ) -> Response:
        _require_same_origin(request)
        try:
            libraries.select_library(library_id)
        except (LookupError, ValueError) as error:
            return render_library_management(
                request,
                error_message=str(error),
                status_code=404,
            )
        return RedirectResponse(request.url_for("library_detail"), status_code=303)

    def render_package_detail(
        request: Request,
        name: str,
        *,
        error_message: str | None = None,
        status_code: int = 200,
    ) -> Response:
        if libraries.current_library is None:
            return RedirectResponse(request.url_for("home"), status_code=303)
        try:
            package = packages.get_package(name)
        except LearningPackageNotFoundError:
            return templates.TemplateResponse(
                request=request,
                name="not_found.html",
                context={"package_name": name},
                status_code=404,
            )

        summary = (
            summaries.get_document_summary(
                package.document_id,
                package.summary_identity_fingerprint,
            )
            if package.summary_identity_fingerprint is not None
            else None
        )
        bank = (
            questions.get_document_bank(
                package.document_id,
                package.question_bank_identity_fingerprint,
            )
            if package.question_bank_identity_fingerprint is not None
            else None
        )
        return templates.TemplateResponse(
            request=request,
            name="package_detail.html",
            context={
                "package": package,
                "question_count": len(bank.questions) if bank is not None else 0,
                "summary": summary,
                "error_message": error_message,
            },
            status_code=status_code,
        )

    @app.get("/package", response_class=HTMLResponse)
    def package_detail(request: Request, name: str) -> Response:
        return render_package_detail(request, name)

    @app.post("/package/rename", response_class=HTMLResponse)
    def rename_package(
        request: Request,
        name: Annotated[str, Form()],
        new_name: Annotated[str, Form()],
    ) -> Response:
        _require_same_origin(request)
        try:
            renamed = libraries.rename_package(name, _validate_package_name(new_name))
        except (LookupError, ValueError) as error:
            return render_package_detail(
                request,
                name,
                error_message=str(error),
                status_code=422,
            )
        location = f"{request.url_for('package_detail')}?{urlencode({'name': renamed.name})}"
        return RedirectResponse(location, status_code=303)

    @app.post("/package/delete", response_class=HTMLResponse)
    def delete_package(
        request: Request,
        name: Annotated[str, Form()],
        confirmation: Annotated[str, Form()],
    ) -> Response:
        _require_same_origin(request)
        try:
            libraries.delete_package(name, confirmation=confirmation)
        except (LookupError, RuntimeError, ValueError) as error:
            return render_package_detail(
                request,
                name,
                error_message=str(error),
                status_code=422,
            )
        return RedirectResponse(request.url_for("library_detail"), status_code=303)

    @app.get("/package/new", response_class=HTMLResponse)
    def package_create(request: Request) -> Response:
        if libraries.current_library is None:
            return RedirectResponse(request.url_for("home"), status_code=303)
        return render_package_create(request)

    @app.post("/package/new", response_class=HTMLResponse)
    async def validate_package_upload(
        request: Request,
        name: Annotated[str, Form()],
        question_count: Annotated[int, Form()],
        pdf: Annotated[UploadFile, File()],
        learning_language: Annotated[LearningLanguage, Form()] = LearningLanguage.SAME_AS_DOCUMENT,
    ) -> Response:
        _require_same_origin(request)
        if libraries.current_library is None:
            raise HTTPException(status_code=409, detail="No library is open")

        try:
            normalized_name = _validate_package_name(name)
            if any(
                package.name.casefold() == normalized_name.casefold()
                for package in packages.list_packages()
            ):
                raise ValueError(f"Learning package already exists: {normalized_name}")
            if any(
                preparation.name.casefold() == normalized_name.casefold()
                for preparation in libraries.list_package_preparations()
            ):
                raise ValueError(f"Learning package already exists: {normalized_name}")
            if not 1 <= question_count <= 50:
                raise ValueError("Question count must be between 1 and 50")
            filename = pdf.filename or ""
            if Path(filename).suffix.casefold() != ".pdf":
                raise ValueError("Choose a PDF file")
            size, prefix, content_sha256 = await _inspect_pdf_upload(pdf)
            if b"%PDF-" not in prefix[:1024]:
                raise ValueError("The uploaded file does not contain a PDF signature")
            await pdf.seek(0)
            libraries.store_package_upload(
                name=normalized_name,
                source_filename=filename,
                question_count=question_count,
                size_bytes=size,
                content_sha256=content_sha256,
                source=pdf.file,
                learning_language=learning_language,
            )
        except ValueError as error:
            return render_package_create(
                request,
                error_message=str(error),
                status_code=422,
            )
        finally:
            await pdf.close()

        return RedirectResponse(request.url_for("library_detail"), status_code=303)

    @app.get("/study", response_class=HTMLResponse)
    def study_question(request: Request, package: str) -> Response:
        if libraries.current_library is None:
            return RedirectResponse(request.url_for("home"), status_code=303)
        due = study.next_due(package, as_of=datetime.now(UTC))
        return templates.TemplateResponse(
            request=request,
            name="study.html",
            context={"due": due, "package_name": package},
        )

    @app.post("/package/retry", response_class=HTMLResponse)
    def retry_package_preparation(
        request: Request,
        preparation_id: Annotated[UUID, Form()],
    ) -> Response:
        _require_same_origin(request)
        if libraries.current_library is None:
            raise HTTPException(status_code=409, detail="No library is open")
        try:
            libraries.retry_package_preparation(preparation_id)
        except ValueError as error:
            raise HTTPException(status_code=409, detail=str(error)) from error
        return RedirectResponse(request.url_for("library_detail"), status_code=303)

    @app.post("/package/preparation/delete", response_class=HTMLResponse)
    def delete_failed_package_preparation(
        request: Request,
        preparation_id: Annotated[UUID, Form()],
    ) -> Response:
        _require_same_origin(request)
        if libraries.current_library is None:
            raise HTTPException(status_code=409, detail="No library is open")
        try:
            libraries.delete_failed_package_preparation(preparation_id)
        except (RuntimeError, ValueError) as error:
            raise HTTPException(status_code=409, detail=str(error)) from error
        return RedirectResponse(request.url_for("library_detail"), status_code=303)

    @app.get("/progress", response_class=HTMLResponse)
    def learning_progress(request: Request, package: str) -> Response:
        if libraries.current_library is None:
            return RedirectResponse(request.url_for("home"), status_code=303)
        report = progress.report(package, as_of=datetime.now(UTC))
        return templates.TemplateResponse(
            request=request,
            name="progress.html",
            context={
                "answered_percent": round(report.answered_rate * 100),
                "correct_percent": round(report.correct_attempt_rate * 100),
                "last_studied": (
                    format_local_datetime(report.last_studied_at)
                    if report.last_studied_at is not None
                    else "Never"
                ),
                "next_due": (
                    format_local_datetime(report.next_due_at)
                    if report.next_due_at is not None
                    else "No review scheduled"
                ),
                "report": report,
            },
        )

    @app.post("/study", response_class=HTMLResponse)
    def submit_study_answer(
        request: Request,
        package: str = Form(),
        question_number: int = Form(),
        answer: str = Form(),
    ) -> HTMLResponse:
        _require_same_origin(request)
        if libraries.current_library is None:
            raise HTTPException(status_code=409, detail="No library is open")
        if not answer.strip():
            raise HTTPException(status_code=422, detail="Study answer must not be blank")

        answered_at = datetime.now(UTC)
        due = study.next_due(package, as_of=answered_at)
        if due is None or due.question.number != question_number:
            raise HTTPException(status_code=409, detail="Study question is no longer due")
        attempt = study.record_answer(
            package,
            question_number,
            answer_text=answer,
            answered_at=answered_at,
        )
        return templates.TemplateResponse(
            request=request,
            name="study_result.html",
            context={
                "attempt": attempt,
                "next_review": format_local_datetime(attempt.resulting_progress.due_at),
                "package_name": package,
            },
        )

    return app


def _require_same_origin(request: Request) -> None:
    """Reject state-changing browser forms sent by another origin."""

    expected_origin = f"{request.url.scheme}://{request.url.netloc}"
    if request.headers.get("origin") != expected_origin:
        raise HTTPException(status_code=403, detail="Cross-origin form submission rejected")


def _validate_package_name(name: str) -> str:
    normalized_name = name.strip()
    if not 1 <= len(normalized_name) <= 100:
        raise ValueError("Package name must contain between 1 and 100 characters")
    if any(ord(character) < 32 for character in normalized_name):
        raise ValueError("Package name must not contain control characters")
    return normalized_name


def _friendly_preparation_error(message: str | None) -> str | None:
    if message is None:
        return None
    if message.startswith("DuplicateDocumentError:"):
        return "This PDF already exists in the library. Remove this failed upload."
    if message.startswith("FileNotFoundError:"):
        return "The uploaded PDF could not be found. Remove this failed upload."
    if "Model response must be valid JSON" in message:
        return "Model processing produced an incomplete response. Retry this package."
    detail = message.split(":", 1)[-1].strip()
    return f"Preparation failed: {detail[:180]}"


async def _inspect_pdf_upload(upload: UploadFile) -> tuple[int, bytes, str]:
    size = 0
    prefix = b""
    digest = sha256()
    while chunk := await upload.read(64 * 1024):
        size += len(chunk)
        digest.update(chunk)
        if len(prefix) < 1024:
            prefix += chunk[: 1024 - len(prefix)]
        if size > MAX_PDF_UPLOAD_BYTES:
            raise ValueError("PDF file must not exceed 25 MiB")
    if size == 0:
        raise ValueError("PDF file must not be empty")
    return size, prefix, digest.hexdigest()
