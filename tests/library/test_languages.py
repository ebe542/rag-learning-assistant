from rag_learning_assistant.library import DocumentLanguage, detect_document_language


def test_detects_german_document_text() -> None:
    text = "Das Dokument ist für die Einführung und erklärt auch die wichtigsten Begriffe."

    assert detect_document_language(text) is DocumentLanguage.GERMAN


def test_detects_english_document_text() -> None:
    text = "This document is an introduction and explains the most important concepts."

    assert detect_document_language(text) is DocumentLanguage.ENGLISH


def test_reports_unknown_for_text_without_language_evidence() -> None:
    assert detect_document_language("Python API 2026") is DocumentLanguage.UNKNOWN


def test_reports_unknown_for_empty_document_text() -> None:
    assert detect_document_language("") is DocumentLanguage.UNKNOWN
