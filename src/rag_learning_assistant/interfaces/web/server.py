"""Run the local browser interface with a fixed security boundary."""

import threading
import webbrowser
from pathlib import Path
from uuid import uuid4

import uvicorn

from rag_learning_assistant.application import (
    AnswerEvaluationService,
    DocumentSummaryCatalog,
    LearningPackageCatalog,
    LearningPackageStudyService,
    LearningProgressService,
    QuestionBankCatalog,
    ReviewScheduler,
    ReviewService,
    StudySessionService,
)
from rag_learning_assistant.generation import (
    HuggingFaceTextGenerator,
    SqliteDocumentSummaryRepository,
)
from rag_learning_assistant.interfaces.web.application import create_app
from rag_learning_assistant.learning import (
    SqliteLearningPackageRepository,
    SqliteQuestionBankRepository,
    SqliteQuestionProgressRepository,
    SqliteStudyAttemptRepository,
)
from rag_learning_assistant.library import SqliteDocumentRepository

LOOPBACK_HOST = "127.0.0.1"


def run_server(
    library_directory: Path,
    port: int,
    *,
    open_browser: bool,
) -> None:
    """Serve the GUI on loopback and optionally open its start page."""

    url = f"http://{LOOPBACK_HOST}:{port}"
    if open_browser:
        browser_timer = threading.Timer(0.75, webbrowser.open, args=(url,))
        browser_timer.daemon = True
        browser_timer.start()

    print(f"RAG Learning Assistant GUI: {url}", flush=True)
    database_path = library_directory / "metadata.sqlite3"
    documents = SqliteDocumentRepository(database_path)
    package_repository = SqliteLearningPackageRepository(database_path)
    progress_repository = SqliteQuestionProgressRepository(database_path)
    attempt_repository = SqliteStudyAttemptRepository(database_path)
    packages = LearningPackageCatalog(package_repository)
    summaries = DocumentSummaryCatalog(
        documents,
        SqliteDocumentSummaryRepository(database_path),
    )
    questions = QuestionBankCatalog(
        documents,
        SqliteQuestionBankRepository(database_path),
    )
    reviewer = ReviewService(
        banks=questions,
        progress=progress_repository,
        scheduler=ReviewScheduler(),
    )
    study = LearningPackageStudyService(
        packages=package_repository,
        sessions=StudySessionService(
            banks=questions,
            reviewer=reviewer,
            attempts=attempt_repository,
            attempt_id_factory=uuid4,
            evaluator=AnswerEvaluationService(HuggingFaceTextGenerator()),
        ),
    )
    progress = LearningProgressService(
        packages=package_repository,
        banks=questions,
        progress=progress_repository,
        attempts=attempt_repository,
    )
    uvicorn.run(
        create_app(
            packages,
            summaries,
            questions,
            study,
            progress,
            library_directory=library_directory,
        ),
        host=LOOPBACK_HOST,
        port=port,
        access_log=False,
    )
