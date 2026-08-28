"""Compose the local FastAPI application."""

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from rag_learning_assistant.learning import LearningPackage

WEB_ROOT = Path(__file__).resolve().parent


class PackageCatalog(Protocol):
    """Supply learning packages without coupling routes to SQLite."""

    def list_packages(self) -> list[LearningPackage]: ...


@dataclass(frozen=True, slots=True)
class PackageListItem:
    """Contain only the learning-package fields rendered by the start page."""

    name: str
    status: str
    status_label: str


def create_app(
    packages: PackageCatalog,
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

    return app
