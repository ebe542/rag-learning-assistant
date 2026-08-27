from fastapi.testclient import TestClient

from rag_learning_assistant.interfaces.web import create_app


def test_home_page_introduces_local_learning_workspace() -> None:
    response = TestClient(create_app()).get("/")

    assert response.status_code == 200
    assert "RAG Learning Assistant" in response.text
    assert "Learning packages" in response.text
    assert "GUI ready" in response.text


def test_static_styles_are_served_locally() -> None:
    response = TestClient(create_app()).get("/static/app.css")

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/css")
    assert ".panel" in response.text
