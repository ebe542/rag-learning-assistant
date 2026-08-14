from pathlib import Path
from uuid import UUID

import pytest

from rag_learning_assistant.application import (
    DocumentNotFoundError,
    DocumentSummaryCatalog,
    DocumentSummaryNotFoundError,
    LibraryService,
)
from rag_learning_assistant.chunking import Chunk
from rag_learning_assistant.generation import (
    Citation,
    PersistedDocumentSummary,
    PromptReference,
)
from rag_learning_assistant.ingestion import Document, Page
from rag_learning_assistant.library import IndexedDocument

DOCUMENT_ID = UUID("aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa")


class RecordingSummaryReader:
    def __init__(
        self,
        found: PersistedDocumentSummary | None = None,
    ) -> None:
        self.found = found
        self.document_ids: list[UUID] = []
        self.find_calls: list[tuple[UUID, str]] = []

    def list_document(self, document_id: UUID) -> list[PersistedDocumentSummary]:
        self.document_ids.append(document_id)
        return []

    def find(
        self,
        document_id: UUID,
        identity_fingerprint: str,
    ) -> PersistedDocumentSummary | None:
        self.find_calls.append((document_id, identity_fingerprint))
        return self.found


class RecordingSummaryRepository:
    def __init__(self, events: list[str]) -> None:
        self.events = events

    def delete_document(self, document_id: UUID) -> int:
        assert document_id == DOCUMENT_ID
        self.events.append("delete summaries")
        return 2


class RecordingQuestionBankRepository:
    def __init__(self, events: list[str]) -> None:
        self.events = events

    def delete_document(self, document_id: UUID) -> int:
        assert document_id == DOCUMENT_ID
        self.events.append("delete question banks")
        return 2


class RecordingDocumentRepository:
    def __init__(
        self,
        document: IndexedDocument,
        events: list[str],
    ) -> None:
        self.document = document
        self.events = events

    def find_by_id(self, document_id: UUID) -> IndexedDocument | None:
        assert document_id == DOCUMENT_ID
        return self.document

    def remove(self, document_id: UUID) -> None:
        assert document_id == DOCUMENT_ID
        self.events.append("remove metadata")

    # Diese Methoden gehören zum vollständigen DocumentRepository-Protokoll,
    # werden bei der Dokumentlöschung aber nicht verwendet.
    def add(self, document: IndexedDocument) -> None:
        raise AssertionError("add must not be called")

    def list_all(self) -> list[IndexedDocument]:
        raise AssertionError("list_all must not be called")

    def find_by_content_hash(
        self,
        content_sha256: str,
    ) -> IndexedDocument | None:
        raise AssertionError("find_by_content_hash must not be called")

    def update(self, document: IndexedDocument) -> None:
        raise AssertionError("update must not be called")


class RecordingIndexer:
    def __init__(self, events: list[str]) -> None:
        self.events = events

    def remove_document(self, document_id: UUID) -> int:
        assert document_id == DOCUMENT_ID
        self.events.append("remove chunks")
        return 2

    # Diese Methoden vervollständigen das DocumentIndexer-Protokoll.
    def index_document(self, document, *, document_id=None):
        raise AssertionError("index_document must not be called")

    def replace_document(self, document, document_id):
        raise AssertionError("replace_document must not be called")


class MismatchedIndexer(RecordingIndexer):
    def remove_document(self, document_id: UUID) -> int:
        assert document_id == DOCUMENT_ID
        self.events.append("remove chunks")
        return 1


class UnexpectedExtractor:
    def extract(self, path):
        raise AssertionError("extract must not be called")


class ReplacementRepository(RecordingDocumentRepository):
    def find_by_content_hash(
        self,
        content_sha256: str,
    ) -> IndexedDocument | None:
        return None

    def update(self, document: IndexedDocument) -> None:
        assert document.id == DOCUMENT_ID
        self.events.append("update metadata")
        self.document = document


class ReplacementExtractor:
    def __init__(
        self,
        document: Document,
        events: list[str],
    ) -> None:
        self.document = document
        self.events = events

    def extract(self, path: Path) -> Document:
        self.events.append("extract document")
        return self.document


class ReplacementIndexer(RecordingIndexer):
    def replace_document(
        self,
        document: Document,
        document_id: UUID,
    ) -> list[Chunk]:
        assert document_id == DOCUMENT_ID
        self.events.append("replace chunks")
        return []


class FailingReplacementIndexer(ReplacementIndexer):
    def replace_document(
        self,
        document: Document,
        document_id: UUID,
    ) -> list[Chunk]:
        assert document_id == DOCUMENT_ID
        self.events.append("replace chunks")
        raise RuntimeError("Index replacement failed")


class MissingDocumentRepository(RecordingDocumentRepository):
    def find_by_id(self, document_id: UUID) -> IndexedDocument | None:
        assert document_id == DOCUMENT_ID
        return None


def test_remove_document_deletes_derived_data_before_catalog_metadata() -> None:
    events: list[str] = []
    document = IndexedDocument(
        id=DOCUMENT_ID,
        source="document.pdf",
        content_sha256="a" * 64,
        page_count=1,
        chunk_count=2,
    )
    service = LibraryService(
        repository=RecordingDocumentRepository(document, events),
        extractor=UnexpectedExtractor(),
        indexer=RecordingIndexer(events),
        derived_data_cleaners=(
            RecordingSummaryRepository(events),
            RecordingQuestionBankRepository(events),
        ),
    )

    removed_document = service.remove_document(DOCUMENT_ID)

    assert removed_document == document
    assert events == [
        "remove chunks",
        "delete summaries",
        "delete question banks",
        "remove metadata",
    ]


def test_remove_document_preserves_derived_data_when_chunk_removal_is_incomplete() -> None:
    events: list[str] = []
    document = IndexedDocument(
        id=DOCUMENT_ID,
        source="document.pdf",
        content_sha256="a" * 64,
        page_count=1,
        chunk_count=2,
    )
    service = LibraryService(
        repository=RecordingDocumentRepository(document, events),
        extractor=UnexpectedExtractor(),
        indexer=MismatchedIndexer(events),
        derived_data_cleaners=(
            RecordingSummaryRepository(events),
            RecordingQuestionBankRepository(events),
        ),
    )

    with pytest.raises(
        RuntimeError,
        match="Removed chunk count does not match document metadata",
    ):
        service.remove_document(DOCUMENT_ID)

    assert events == ["remove chunks"]


def test_replace_document_deletes_derived_data_before_updating_metadata(
    tmp_path: Path,
) -> None:
    events: list[str] = []
    original = IndexedDocument(
        id=DOCUMENT_ID,
        source="old.pdf",
        content_sha256="a" * 64,
        page_count=1,
        chunk_count=2,
    )
    replacement = Document(
        source="new.pdf",
        pages=(Page(1, "Replacement content", "new.pdf"),),
    )
    replacement_path = tmp_path / "new.pdf"
    replacement_path.write_bytes(b"replacement PDF contents")

    service = LibraryService(
        repository=ReplacementRepository(original, events),
        extractor=ReplacementExtractor(replacement, events),
        indexer=ReplacementIndexer(events),
        derived_data_cleaners=(
            RecordingSummaryRepository(events),
            RecordingQuestionBankRepository(events),
        ),
    )

    replaced_document = service.replace_document(
        DOCUMENT_ID,
        replacement_path,
    )

    assert replaced_document.source == "new.pdf"
    assert events == [
        "extract document",
        "replace chunks",
        "delete summaries",
        "delete question banks",
        "update metadata",
    ]


def test_replace_document_preserves_derived_data_when_indexing_fails(
    tmp_path: Path,
) -> None:
    events: list[str] = []
    original = IndexedDocument(
        id=DOCUMENT_ID,
        source="old.pdf",
        content_sha256="a" * 64,
        page_count=1,
        chunk_count=2,
    )
    replacement = Document(
        source="new.pdf",
        pages=(Page(1, "Replacement content", "new.pdf"),),
    )
    replacement_path = tmp_path / "new.pdf"
    replacement_path.write_bytes(b"replacement PDF contents")

    service = LibraryService(
        repository=ReplacementRepository(original, events),
        extractor=ReplacementExtractor(replacement, events),
        indexer=FailingReplacementIndexer(events),
        derived_data_cleaners=(
            RecordingSummaryRepository(events),
            RecordingQuestionBankRepository(events),
        ),
    )

    with pytest.raises(RuntimeError, match="Index replacement failed"):
        service.replace_document(DOCUMENT_ID, replacement_path)

    assert events == [
        "extract document",
        "replace chunks",
    ]


def test_summary_catalog_lists_saved_versions_for_known_document() -> None:
    events: list[str] = []
    document = IndexedDocument(
        id=DOCUMENT_ID,
        source="document.pdf",
        content_sha256="a" * 64,
        page_count=1,
        chunk_count=2,
    )
    summaries = RecordingSummaryReader()
    catalog = DocumentSummaryCatalog(
        documents=RecordingDocumentRepository(document, events),
        summaries=summaries,
    )

    result = catalog.list_document_summaries(DOCUMENT_ID)

    assert result == []
    assert summaries.document_ids == [DOCUMENT_ID]


def test_summary_catalog_rejects_unknown_document_before_reading_summaries() -> None:
    events: list[str] = []
    placeholder = IndexedDocument(
        id=DOCUMENT_ID,
        source="document.pdf",
        content_sha256="a" * 64,
        page_count=1,
        chunk_count=2,
    )
    summaries = RecordingSummaryReader()
    catalog = DocumentSummaryCatalog(
        documents=MissingDocumentRepository(placeholder, events),
        summaries=summaries,
    )

    with pytest.raises(
        DocumentNotFoundError,
        match=f"Document does not exist: {DOCUMENT_ID}",
    ):
        catalog.list_document_summaries(DOCUMENT_ID)

    assert summaries.document_ids == []


def test_summary_catalog_reports_unknown_summary_identity() -> None:
    events: list[str] = []
    document = IndexedDocument(
        id=DOCUMENT_ID,
        source="document.pdf",
        content_sha256="a" * 64,
        page_count=1,
        chunk_count=2,
    )
    summaries = RecordingSummaryReader()
    catalog = DocumentSummaryCatalog(
        documents=RecordingDocumentRepository(document, events),
        summaries=summaries,
    )
    identity_fingerprint = "b" * 64

    with pytest.raises(
        DocumentSummaryNotFoundError,
        match="Stored document summary does not exist",
    ):
        catalog.get_document_summary(
            DOCUMENT_ID,
            identity_fingerprint,
        )

    assert summaries.find_calls == [
        (DOCUMENT_ID, identity_fingerprint),
    ]


def test_summary_catalog_returns_exact_saved_identity() -> None:
    events: list[str] = []
    document = IndexedDocument(
        id=DOCUMENT_ID,
        source="document.pdf",
        content_sha256="a" * 64,
        page_count=1,
        chunk_count=2,
    )
    summary = PersistedDocumentSummary(
        document_id=DOCUMENT_ID,
        identity_fingerprint="b" * 64,
        source="document.pdf",
        text="Saved summary.",
        citations=(
            Citation(
                number=1,
                source="document.pdf",
                page_number=1,
                chunk_index=0,
                excerpt="Supporting passage.",
            ),
        ),
        prompt_references=(
            PromptReference(
                name="summarization.reduce",
                version=2,
                fingerprint="c" * 64,
            ),
        ),
    )
    summaries = RecordingSummaryReader(found=summary)
    catalog = DocumentSummaryCatalog(
        documents=RecordingDocumentRepository(document, events),
        summaries=summaries,
    )

    result = catalog.get_document_summary(
        DOCUMENT_ID,
        summary.identity_fingerprint,
    )

    assert result == summary
    assert summaries.find_calls == [
        (DOCUMENT_ID, summary.identity_fingerprint),
    ]
