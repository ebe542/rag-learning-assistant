"""Sentence Transformers adapter for local text embeddings."""

from collections.abc import Iterable, Sequence
from typing import Protocol, cast

from rag_learning_assistant.retrieval.embeddings import Embedding

DEFAULT_MODEL_NAME = "intfloat/multilingual-e5-small"
DEFAULT_MODEL_REVISION = "614241f622f53c4eeff9890bdc4f31cfecc418b3"


class SentenceTransformerModel(Protocol):
    """Subset of the Sentence Transformers API used by the adapter."""

    def encode(
        self,
        sentences: Sequence[str],
        *,
        normalize_embeddings: bool,
    ) -> Iterable[Iterable[float]]:
        """Encode texts as vectors."""
        ...


class SentenceTransformerEmbedder:
    """Create E5-compatible document and query embeddings."""

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL_NAME,
        model_revision: str = DEFAULT_MODEL_REVISION,
        model: SentenceTransformerModel | None = None,
    ) -> None:
        self.model_name = model_name
        self.model_revision = model_revision
        self._model = model

    def embed_documents(
        self,
        texts: Sequence[str],
    ) -> list[Embedding]:
        """Embed document texts using the E5 passage prefix."""

        prefixed_texts = [f"passage: {text}" for text in texts]
        return self._encode(prefixed_texts)

    def embed_query(self, text: str) -> Embedding:
        """Embed a search query using the E5 query prefix."""

        embeddings = self._encode([f"query: {text}"])
        return embeddings[0]

    def _encode(self, texts: Sequence[str]) -> list[Embedding]:
        vectors = self._get_model().encode(
            texts,
            normalize_embeddings=True,
        )
        return [tuple(float(value) for value in vector) for vector in vectors]

    def _get_model(self) -> SentenceTransformerModel:
        if self._model is None:
            self._model = self._load_model()

        return self._model

    def _load_model(self) -> SentenceTransformerModel:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise RuntimeError(
                "Sentence Transformers is not installed. "
                "Install the embeddings optional dependency."
            ) from exc

        model = SentenceTransformer(
            self.model_name,
            revision=self.model_revision,
        )
        return cast(SentenceTransformerModel, model)
