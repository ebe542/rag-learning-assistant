from pathlib import Path
from uuid import UUID

from rag_learning_assistant.application import (
    BatchImportService,
    DuplicateDocumentError,
    ImportStatus,
)
from rag_learning_assistant.library import IndexedDocument


class RecordingLibraryService:
    def __init__(
        self,
        documents: dict[Path, IndexedDocument],
    ) -> None:
        self.documents = documents
        self.paths: list[Path] = []

    def add_document(self, path: Path) -> IndexedDocument:
        self.paths.append(path)
        return self.documents[path]


def test_add_documents_processes_every_path_in_order() -> None:
    first_path = Path("first.pdf")
    second_path = Path("second.pdf")
    first_document = IndexedDocument(
        id=UUID("11111111-1111-1111-1111-111111111111"),
        source="first.pdf",
        content_sha256="a" * 64,
        page_count=10,
        chunk_count=20,
    )
    second_document = IndexedDocument(
        id=UUID("22222222-2222-2222-2222-222222222222"),
        source="second.pdf",
        content_sha256="b" * 64,
        page_count=15,
        chunk_count=30,
    )
    library = RecordingLibraryService(
        {
            first_path: first_document,
            second_path: second_document,
        }
    )
    service = BatchImportService(library)

    outcomes = service.add_documents([first_path, second_path])

    assert library.paths == [first_path, second_path]
    assert [outcome.status for outcome in outcomes] == [
        ImportStatus.ADDED,
        ImportStatus.ADDED,
    ]
    assert [outcome.document for outcome in outcomes] == [
        first_document,
        second_document,
    ]


def test_duplicate_document_is_skipped() -> None:
    duplicate_path = Path("duplicate.pdf")

    class DuplicateLibrary:
        def add_document(
            self,
            path: Path,
        ) -> IndexedDocument:
            raise DuplicateDocumentError("Document content is already indexed as original.pdf")

    service = BatchImportService(DuplicateLibrary())

    outcomes = service.add_documents([duplicate_path])

    assert len(outcomes) == 1
    assert outcomes[0].path == duplicate_path
    assert outcomes[0].status is ImportStatus.SKIPPED
    assert outcomes[0].document is None
    assert outcomes[0].message == ("Document content is already indexed as original.pdf")


def test_failed_document_does_not_stop_remaining_imports() -> None:
    broken_path = Path("broken.pdf")
    valid_path = Path("valid.pdf")
    valid_document = IndexedDocument(
        id=UUID("12345678-1234-5678-1234-567812345678"),
        source="valid.pdf",
        content_sha256="a" * 64,
        page_count=10,
        chunk_count=20,
    )

    class PartiallyFailingLibrary:
        def __init__(self) -> None:
            self.paths: list[Path] = []

        def add_document(
            self,
            path: Path,
        ) -> IndexedDocument:
            self.paths.append(path)

            if path == broken_path:
                raise ValueError("Could not open PDF")

            return valid_document

    library = PartiallyFailingLibrary()
    service = BatchImportService(library)

    outcomes = service.add_documents([broken_path, valid_path])

    assert library.paths == [broken_path, valid_path]
    assert outcomes[0].status is ImportStatus.FAILED
    assert outcomes[0].document is None
    assert outcomes[0].message == "Could not open PDF"
    assert outcomes[1].status is ImportStatus.ADDED
    assert outcomes[1].document == valid_document
    assert outcomes[1].message is None
