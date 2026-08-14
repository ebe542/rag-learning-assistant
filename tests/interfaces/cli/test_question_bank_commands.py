import json
from pathlib import Path
from uuid import UUID

from rag_learning_assistant.application import (
    DocumentSummaryCatalog,
    QuestionBankService,
)
from rag_learning_assistant.application.question_bank import (
    QUESTION_BANK_PROMPT,
)
from rag_learning_assistant.chunking import TextChunker
from rag_learning_assistant.generation import (
    Citation,
    HuggingFaceTextGenerator,
    PersistedDocumentSummary,
    PromptReference,
    SqliteDocumentSummaryRepository,
)
from rag_learning_assistant.generation.huggingface import (
    QUESTION_JSON_REPAIR_PROMPT,
    QUESTION_SYSTEM_PROMPT,
)
from rag_learning_assistant.interfaces.cli import (
    commands,
    entrypoint,
)
from rag_learning_assistant.interfaces.cli.parser import (
    DEFAULT_QUESTION_COUNT,
    DEFAULT_QUESTION_MAX_NEW_TOKENS,
    build_parser,
)
from rag_learning_assistant.learning import (
    QuestionBank,
    SqliteQuestionBankRepository,
    StudyQuestion,
)
from rag_learning_assistant.library import (
    SqliteDocumentRepository,
)


def test_question_bank_builder_uses_library_database_and_lazy_generator(
    tmp_path: Path,
) -> None:
    service = commands.build_question_bank_service(
        index_directory=tmp_path,
        max_new_tokens=256,
    )

    assert isinstance(service, QuestionBankService)
    assert isinstance(service.summaries, DocumentSummaryCatalog)
    assert isinstance(service.generator, HuggingFaceTextGenerator)
    assert service.generator.max_new_tokens == 256
    assert isinstance(
        service.banks,
        SqliteQuestionBankRepository,
    )
    assert service.banks.database_path == tmp_path / "metadata.sqlite3"


def test_question_bank_builder_versions_all_generation_inputs(
    tmp_path: Path,
) -> None:
    service = commands.build_question_bank_service(
        index_directory=tmp_path,
        max_new_tokens=256,
    )
    summary = PersistedDocumentSummary(
        document_id=UUID(
            "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa",
        ),
        identity_fingerprint="b" * 64,
        source="course.pdf",
        text="Grounded summary.",
        citations=(
            Citation(
                number=1,
                source="course.pdf",
                page_number=1,
                chunk_index=0,
                excerpt="Supporting passage.",
            ),
        ),
        prompt_references=(
            PromptReference(
                name="summarization.reduce",
                version=4,
                fingerprint="c" * 64,
            ),
        ),
    )

    identity = service.identity_factory(summary, 5)

    assert identity.prompt_references == (
        QUESTION_BANK_PROMPT.reference,
        QUESTION_SYSTEM_PROMPT.reference,
        QUESTION_JSON_REPAIR_PROMPT.reference,
    )
    assert identity.question_count == 5
    assert identity.max_new_tokens == 256
    assert identity.summary_identity_fingerprint == "b" * 64


def test_parser_accepts_question_generate_command() -> None:
    args = build_parser().parse_args(
        [
            "question-generate",
            "local-data/indexes/library",
            "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa",
            "b" * 64,
            "--count",
            "8",
            "--max-new-tokens",
            "640",
            "--force",
        ]
    )

    assert args.command == "question-generate"
    assert args.index_dir == Path("local-data/indexes/library")
    assert args.document_id == UUID(
        "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa",
    )
    assert args.summary_identity_fingerprint == "b" * 64
    assert args.count == 8
    assert args.max_new_tokens == 640
    assert args.force is True


def test_question_generate_parser_uses_documented_defaults() -> None:
    args = build_parser().parse_args(
        [
            "question-generate",
            "local-data/indexes/library",
            "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa",
            "b" * 64,
        ]
    )

    assert args.count == DEFAULT_QUESTION_COUNT
    assert args.max_new_tokens == DEFAULT_QUESTION_MAX_NEW_TOKENS
    assert args.force is False


class RecordingQuestionBankService:
    def __init__(self, bank: QuestionBank) -> None:
        self.bank = bank
        self.calls: list[tuple[UUID, str, int, bool]] = []

    def generate(
        self,
        document_id: UUID,
        summary_identity_fingerprint: str,
        *,
        question_count: int,
        force: bool = False,
    ) -> QuestionBank:
        self.calls.append(
            (
                document_id,
                summary_identity_fingerprint,
                question_count,
                force,
            )
        )
        return self.bank


def test_run_question_generate_outputs_grounded_bank_as_json(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    document_id = UUID("aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa")
    summary_identity = "b" * 64
    bank_identity = "d" * 64
    citation = Citation(
        number=1,
        source="course.pdf",
        page_number=4,
        chunk_index=7,
        excerpt="Embeddings represent text as numeric vectors.",
    )
    prompt = PromptReference(
        name="question-bank.generate",
        version=1,
        fingerprint="e" * 64,
    )
    bank = QuestionBank(
        document_id=document_id,
        identity_fingerprint=bank_identity,
        source="course.pdf",
        questions=(
            StudyQuestion(
                number=1,
                text="What is an embedding?",
                expected_answer=("A numeric representation of text."),
                citations=(citation,),
            ),
        ),
        prompt_references=(prompt,),
    )
    service = RecordingQuestionBankService(bank)
    builder_calls: list[tuple[Path, int]] = []

    def fake_build_question_bank_service(
        index_directory: Path,
        max_new_tokens: int,
    ) -> RecordingQuestionBankService:
        builder_calls.append(
            (index_directory, max_new_tokens),
        )
        return service

    monkeypatch.setattr(
        commands,
        "build_question_bank_service",
        fake_build_question_bank_service,
    )

    exit_code = commands.run_question_generate(
        index_directory=tmp_path,
        document_id=document_id,
        summary_identity_fingerprint=summary_identity,
        question_count=1,
        max_new_tokens=256,
        force=True,
    )

    assert exit_code == 0
    assert builder_calls == [(tmp_path, 256)]
    assert service.calls == [
        (document_id, summary_identity, 1, True),
    ]
    assert json.loads(capsys.readouterr().out) == {
        "index_directory": str(tmp_path),
        "document_id": str(document_id),
        "summary_identity_fingerprint": summary_identity,
        "question_bank_identity_fingerprint": bank_identity,
        "source": "course.pdf",
        "questions": [
            {
                "number": 1,
                "text": "What is an embedding?",
                "expected_answer": ("A numeric representation of text."),
                "citations": [
                    {
                        "number": 1,
                        "source": "course.pdf",
                        "page_number": 4,
                        "chunk_index": 7,
                        "excerpt": ("Embeddings represent text as numeric vectors."),
                    }
                ],
            }
        ],
        "prompts": [
            {
                "name": "question-bank.generate",
                "version": 1,
                "fingerprint": "e" * 64,
            }
        ],
    }


def test_entrypoint_dispatches_question_generate_command(
    monkeypatch,
) -> None:
    document_id = UUID("aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa")
    summary_identity = "b" * 64
    calls: list[tuple[Path, UUID, str, int, int, bool]] = []

    def fake_run_question_generate(
        index_directory: Path,
        document_id: UUID,
        summary_identity_fingerprint: str,
        question_count: int,
        max_new_tokens: int,
        force: bool,
    ) -> int:
        calls.append(
            (
                index_directory,
                document_id,
                summary_identity_fingerprint,
                question_count,
                max_new_tokens,
                force,
            )
        )
        return 0

    monkeypatch.setattr(
        entrypoint.commands,
        "run_question_generate",
        fake_run_question_generate,
    )
    monkeypatch.setattr(
        entrypoint,
        "validate_library_directory",
        lambda path: None,
    )

    exit_code = entrypoint.main(
        [
            "question-generate",
            "local-data/indexes/library",
            str(document_id),
            summary_identity,
            "--count",
            "8",
            "--max-new-tokens",
            "640",
            "--force",
        ]
    )

    assert exit_code == 0
    assert calls == [
        (
            Path("local-data/indexes/library"),
            document_id,
            summary_identity,
            8,
            640,
            True,
        )
    ]


def test_question_bank_catalog_builder_uses_library_database(
    tmp_path: Path,
) -> None:
    catalog = commands.build_question_bank_catalog(tmp_path)

    assert isinstance(
        catalog.documents,
        SqliteDocumentRepository,
    )
    assert catalog.documents.database_path == (tmp_path / "metadata.sqlite3")
    assert isinstance(
        catalog.banks,
        SqliteQuestionBankRepository,
    )
    assert catalog.banks.database_path == (tmp_path / "metadata.sqlite3")


def test_parser_accepts_question_list_command() -> None:
    args = build_parser().parse_args(
        [
            "question-list",
            "local-data/indexes/library",
            "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa",
        ]
    )

    assert args.command == "question-list"
    assert args.index_dir == Path("local-data/indexes/library")
    assert args.document_id == UUID(
        "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa",
    )


def test_parser_accepts_question_show_command() -> None:
    args = build_parser().parse_args(
        [
            "question-show",
            "local-data/indexes/library",
            "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa",
            "d" * 64,
        ]
    )

    assert args.command == "question-show"
    assert args.index_dir == Path("local-data/indexes/library")
    assert args.document_id == UUID(
        "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa",
    )
    assert args.identity_fingerprint == "d" * 64


class RecordingQuestionBankCatalog:
    def __init__(self, banks: list[QuestionBank]) -> None:
        self.banks = banks
        self.list_calls: list[UUID] = []
        self.get_calls: list[tuple[UUID, str]] = []

    def list_document_banks(
        self,
        document_id: UUID,
    ) -> list[QuestionBank]:
        self.list_calls.append(document_id)
        return list(self.banks)

    def get_document_bank(
        self,
        document_id: UUID,
        identity_fingerprint: str,
    ) -> QuestionBank:
        self.get_calls.append(
            (document_id, identity_fingerprint),
        )
        return self.banks[0]


def test_run_question_list_outputs_bank_metadata_as_json(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    document_id = UUID("aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa")
    prompt = PromptReference(
        name="question-bank.generate",
        version=1,
        fingerprint="e" * 64,
    )
    bank = QuestionBank(
        document_id=document_id,
        identity_fingerprint="d" * 64,
        source="course.pdf",
        questions=(
            StudyQuestion(
                number=1,
                text="What is an embedding?",
                expected_answer="A numeric representation.",
                citations=(
                    Citation(
                        number=1,
                        source="course.pdf",
                        page_number=4,
                        chunk_index=7,
                        excerpt="Supporting passage.",
                    ),
                ),
            ),
        ),
        prompt_references=(prompt,),
    )
    catalog = RecordingQuestionBankCatalog([bank])
    monkeypatch.setattr(
        commands,
        "build_question_bank_catalog",
        lambda index_directory: catalog,
    )

    exit_code = commands.run_question_list(
        tmp_path,
        document_id,
    )

    assert exit_code == 0
    assert catalog.list_calls == [document_id]
    assert json.loads(capsys.readouterr().out) == {
        "index_directory": str(tmp_path),
        "document_id": str(document_id),
        "question_banks": [
            {
                "identity_fingerprint": "d" * 64,
                "source": "course.pdf",
                "question_count": 1,
                "prompts": [
                    {
                        "name": "question-bank.generate",
                        "version": 1,
                        "fingerprint": "e" * 64,
                    }
                ],
            }
        ],
    }


def test_run_question_show_outputs_complete_bank_as_json(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    document_id = UUID("aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa")
    identity_fingerprint = "d" * 64
    citation = Citation(
        number=1,
        source="course.pdf",
        page_number=4,
        chunk_index=7,
        excerpt="Supporting passage.",
    )
    prompt = PromptReference(
        name="question-bank.generate",
        version=1,
        fingerprint="e" * 64,
    )
    bank = QuestionBank(
        document_id=document_id,
        identity_fingerprint=identity_fingerprint,
        source="course.pdf",
        questions=(
            StudyQuestion(
                number=1,
                text="What is an embedding?",
                expected_answer="A numeric representation.",
                citations=(citation,),
            ),
        ),
        prompt_references=(prompt,),
    )
    catalog = RecordingQuestionBankCatalog([bank])
    monkeypatch.setattr(
        commands,
        "build_question_bank_catalog",
        lambda index_directory: catalog,
    )

    exit_code = commands.run_question_show(
        tmp_path,
        document_id,
        identity_fingerprint,
    )

    assert exit_code == 0
    assert catalog.get_calls == [
        (document_id, identity_fingerprint),
    ]
    assert json.loads(capsys.readouterr().out) == {
        "index_directory": str(tmp_path),
        "document_id": str(document_id),
        "identity_fingerprint": identity_fingerprint,
        "source": "course.pdf",
        "questions": [
            {
                "number": 1,
                "text": "What is an embedding?",
                "expected_answer": "A numeric representation.",
                "citations": [
                    {
                        "number": 1,
                        "source": "course.pdf",
                        "page_number": 4,
                        "chunk_index": 7,
                        "excerpt": "Supporting passage.",
                    }
                ],
            }
        ],
        "prompts": [
            {
                "name": "question-bank.generate",
                "version": 1,
                "fingerprint": "e" * 64,
            }
        ],
    }


def test_entrypoint_dispatches_question_list_command(
    monkeypatch,
) -> None:
    document_id = UUID("aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa")
    calls: list[tuple[Path, UUID]] = []

    def fake_run_question_list(
        index_directory: Path,
        document_id: UUID,
    ) -> int:
        calls.append((index_directory, document_id))
        return 0

    monkeypatch.setattr(
        entrypoint.commands,
        "run_question_list",
        fake_run_question_list,
    )
    monkeypatch.setattr(
        entrypoint,
        "validate_library_directory",
        lambda path: None,
    )

    exit_code = entrypoint.main(
        [
            "question-list",
            "local-data/indexes/library",
            str(document_id),
        ]
    )

    assert exit_code == 0
    assert calls == [
        (
            Path("local-data/indexes/library"),
            document_id,
        )
    ]


def test_entrypoint_dispatches_question_show_command(
    monkeypatch,
) -> None:
    document_id = UUID("aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa")
    identity_fingerprint = "d" * 64
    calls: list[tuple[Path, UUID, str]] = []

    def fake_run_question_show(
        index_directory: Path,
        document_id: UUID,
        identity_fingerprint: str,
    ) -> int:
        calls.append(
            (
                index_directory,
                document_id,
                identity_fingerprint,
            )
        )
        return 0

    monkeypatch.setattr(
        entrypoint.commands,
        "run_question_show",
        fake_run_question_show,
    )
    monkeypatch.setattr(
        entrypoint,
        "validate_library_directory",
        lambda path: None,
    )

    exit_code = entrypoint.main(
        [
            "question-show",
            "local-data/indexes/library",
            str(document_id),
            identity_fingerprint,
        ]
    )

    assert exit_code == 0
    assert calls == [
        (
            Path("local-data/indexes/library"),
            document_id,
            identity_fingerprint,
        )
    ]


def test_library_builder_registers_all_derived_data_cleaners(
    tmp_path: Path,
    monkeypatch,
) -> None:
    fake_indexer = object()

    def fake_build_persistent_document_search(
        chunker: TextChunker,
        index_directory: Path,
    ) -> object:
        return fake_indexer

    monkeypatch.setattr(
        commands,
        "build_persistent_document_search",
        fake_build_persistent_document_search,
    )

    service = commands.build_library_service(
        TextChunker(max_chars=1000, overlap_chars=100),
        tmp_path,
    )

    assert len(service.derived_data_cleaners) == 2
    assert isinstance(
        service.derived_data_cleaners[0],
        SqliteDocumentSummaryRepository,
    )
    assert isinstance(
        service.derived_data_cleaners[1],
        SqliteQuestionBankRepository,
    )
