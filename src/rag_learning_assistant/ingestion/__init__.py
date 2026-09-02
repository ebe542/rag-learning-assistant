"""Document ingestion and its data models."""

from rag_learning_assistant.ingestion.models import Document, Page, has_machine_readable_text
from rag_learning_assistant.ingestion.ocr import PageOcr
from rag_learning_assistant.ingestion.pdf import PdfExtractor
from rag_learning_assistant.ingestion.tesseract import (
    DEFAULT_OCR_LANGUAGES,
    TesseractPageOcr,
)

__all__ = [
    "Document",
    "DEFAULT_OCR_LANGUAGES",
    "Page",
    "PageOcr",
    "PdfExtractor",
    "TesseractPageOcr",
    "has_machine_readable_text",
]
