"""Learner-selected output languages for generated material."""

from enum import StrEnum

from rag_learning_assistant.library import DocumentLanguage


class LearningLanguage(StrEnum):
    """Select the language used for generated learning material."""

    SAME_AS_DOCUMENT = "same"
    GERMAN = "de"
    ENGLISH = "en"

    def resolve(self, document_language: DocumentLanguage) -> DocumentLanguage:
        """Resolve the requested output language against the source language."""

        if self is LearningLanguage.SAME_AS_DOCUMENT:
            return document_language
        return DocumentLanguage(self.value)
