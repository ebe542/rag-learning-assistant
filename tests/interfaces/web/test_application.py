from pathlib import Path
from uuid import UUID

from fastapi.testclient import TestClient

from rag_learning_assistant.interfaces.web import create_app
from rag_learning_assistant.learning import LearningPackage, LearningPackageStatus


class StubPackageCatalog:
    def __init__(self, packages: list[LearningPackage]) -> None:
        self.packages = packages

    def list_packages(self) -> list[LearningPackage]:
        return self.packages


def build_client(packages: list[LearningPackage] | None = None) -> TestClient:
    return TestClient(
        create_app(
            StubPackageCatalog(packages or []),
            library_directory=Path("personal-library"),
        )
    )


def test_home_page_introduces_local_learning_workspace() -> None:
    response = build_client().get("/")

    assert response.status_code == 200
    assert "RAG Learning Assistant" in response.text
    assert "Learning packages" in response.text
    assert "No learning packages yet" in response.text
    assert "personal-library" in response.text


def test_home_page_lists_packages_with_their_preparation_status() -> None:
    package = LearningPackage(
        id=UUID("aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa"),
        name="Python Basics",
        document_id=UUID("bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb"),
        status=LearningPackageStatus.READY,
        summary_identity_fingerprint="a" * 64,
        question_bank_identity_fingerprint="b" * 64,
    )

    response = build_client([package]).get("/")

    assert response.status_code == 200
    assert "Python Basics" in response.text
    assert "Ready" in response.text
    assert "No learning packages yet" not in response.text


def test_static_styles_are_served_locally() -> None:
    response = build_client().get("/static/app.css")

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/css")
    assert ".panel" in response.text
