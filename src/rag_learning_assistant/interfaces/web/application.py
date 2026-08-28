"""Compose the local FastAPI application."""

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol
from uuid import UUID

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from rag_learning_assistant.application import LearningPackageNotFoundError
from rag_learning_assistant.generation import PersistedDocumentSummary
from rag_learning_assistant.learning import LearningPackage, QuestionBank

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


@dataclass(frozen=True, slots=True)
class PackageListItem:
    """Contain only the learning-package fields rendered by the start page."""

    name: str
    status: str
    status_label: str


def create_app(
    packages: PackageCatalog,
    summaries: SummaryCatalog,
    questions: QuestionCatalog,
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

    return app
