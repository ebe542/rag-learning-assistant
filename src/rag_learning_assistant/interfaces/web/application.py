"""Compose the local FastAPI application."""

from dataclasses import dataclass
from datetime import UTC, datetime, tzinfo
from pathlib import Path
from typing import Protocol
from uuid import UUID

from fastapi import FastAPI, Form, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.middleware.trustedhost import TrustedHostMiddleware

from rag_learning_assistant.application import (
    LearningPackageNotFoundError,
    LearningProgressReport,
)
from rag_learning_assistant.application.review import DueQuestion
from rag_learning_assistant.generation import PersistedDocumentSummary
from rag_learning_assistant.learning import LearningPackage, QuestionBank, StudyAttempt

WEB_ROOT = Path(__file__).resolve().parent


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


@dataclass(frozen=True, slots=True)
class PackageListItem:
    """Contain only the learning-package fields rendered by the start page."""

    name: str
    status: str
    status_label: str


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
    library_directory: Path,
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
    templates = Jinja2Templates(directory=WEB_ROOT / "templates")

    @app.get("/", response_class=HTMLResponse)
    def home(request: Request) -> HTMLResponse:
        package_items = tuple(
            PackageListItem(
                name=package.name,
                status=package.status.value,
                status_label=package.status.value.capitalize(),
            )
            for package in packages.list_packages()
        )
        return templates.TemplateResponse(
            request=request,
            name="home.html",
            context={
                "library_directory": str(library_directory),
                "packages": package_items,
            },
        )

    @app.get("/package", response_class=HTMLResponse)
    def package_detail(request: Request, name: str) -> HTMLResponse:
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
            },
        )

    @app.get("/study", response_class=HTMLResponse)
    def study_question(request: Request, package: str) -> HTMLResponse:
        due = study.next_due(package, as_of=datetime.now(UTC))
        return templates.TemplateResponse(
            request=request,
            name="study.html",
            context={"due": due, "package_name": package},
        )

    @app.get("/progress", response_class=HTMLResponse)
    def learning_progress(request: Request, package: str) -> HTMLResponse:
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
        expected_origin = f"{request.url.scheme}://{request.url.netloc}"
        if request.headers.get("origin") != expected_origin:
            raise HTTPException(status_code=403, detail="Cross-origin form submission rejected")
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
