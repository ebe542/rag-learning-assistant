import sys
from collections.abc import Sequence
from types import SimpleNamespace

from rag_learning_assistant.retrieval import SentenceTransformerEmbedder


class FakeSentenceTransformer:
    """Record model input and return controlled vectors."""

    def __init__(self) -> None:
        self.calls: list[tuple[list[str], bool]] = []

    def encode(
        self,
        sentences: Sequence[str],
        *,
        normalize_embeddings: bool,
    ) -> list[list[float]]:
        self.calls.append((list(sentences), normalize_embeddings))
        return [[float(index), 1.0] for index, _ in enumerate(sentences)]


def test_embed_documents_adds_passage_prefixes() -> None:
    model = FakeSentenceTransformer()
    embedder = SentenceTransformerEmbedder(
        model_name="intfloat/multilingual-e5-small",
        model=model,
    )

    embeddings = embedder.embed_documents(["Python functions", "Python classes"])

    assert model.calls == [
        (
            [
                "passage: Python functions",
                "passage: Python classes",
            ],
            True,
        )
    ]
    assert embeddings == [(0.0, 1.0), (1.0, 1.0)]


def test_embed_query_adds_query_prefix() -> None:
    model = FakeSentenceTransformer()
    embedder = SentenceTransformerEmbedder(
        model_name="intfloat/multilingual-e5-small",
        model=model,
    )

    embedding = embedder.embed_query("How do functions work?")

    assert model.calls == [
        (
            ["query: How do functions work?"],
            True,
        )
    ]
    assert embedding == (0.0, 1.0)


def test_model_is_loaded_lazily_and_reused(monkeypatch) -> None:
    model = FakeSentenceTransformer()
    load_calls: list[str] = []
    embedder = SentenceTransformerEmbedder(model_name="intfloat/multilingual-e5-small")

    def load_model() -> FakeSentenceTransformer:
        load_calls.append("loaded")
        return model

    monkeypatch.setattr(embedder, "_load_model", load_model)

    assert load_calls == []

    embedder.embed_query("First query")
    embedder.embed_query("Second query")

    assert load_calls == ["loaded"]


def test_model_loading_uses_pinned_revision(monkeypatch) -> None:
    model = FakeSentenceTransformer()
    loaded_with: dict[str, str] = {}

    def load_model(
        model_name: str,
        *,
        revision: str,
    ) -> FakeSentenceTransformer:
        loaded_with["model_name"] = model_name
        loaded_with["revision"] = revision
        return model

    fake_module = SimpleNamespace(SentenceTransformer=load_model)
    monkeypatch.setitem(sys.modules, "sentence_transformers", fake_module)

    embedder = SentenceTransformerEmbedder(
        model_name="intfloat/multilingual-e5-small",
        model_revision="614241f622f53c4eeff9890bdc4f31cfecc418b3",
    )

    embedder.embed_query("What is Python?")

    assert loaded_with == {
        "model_name": "intfloat/multilingual-e5-small",
        "revision": "614241f622f53c4eeff9890bdc4f31cfecc418b3",
    }
