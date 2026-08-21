"""Command-line argument dispatch."""

from collections.abc import Sequence
from contextlib import suppress

from dotenv import load_dotenv

from rag_learning_assistant.application import (
    DocumentNotFoundError,
    DocumentSummaryNotFoundError,
    DuplicateDocumentError,
    LearningPackageNotFoundError,
    LearningPackageNotReadyError,
    QuestionBankNotFoundError,
    StudyQuestionNotFoundError,
)
from rag_learning_assistant.chunking import TextChunker
from rag_learning_assistant.ingestion import PdfExtractor
from rag_learning_assistant.interfaces.cli import commands
from rag_learning_assistant.interfaces.cli.parser import (
    DEFAULT_MAX_CHARS,
    DEFAULT_OVERLAP_CHARS,
    build_parser,
    validate_existing_index_directory,
    validate_index_directory,
    validate_library_directory,
)


def main(argv: Sequence[str] | None = None) -> int:
    """Parse arguments and execute the selected command."""

    # Environment loading is optional. A missing or unreadable .env must not
    # prevent commands that do not need Hugging Face authentication.
    with suppress(Exception):
        load_dotenv()

    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "prepare":
        try:
            validate_index_directory(args.library)
        except ValueError as exc:
            parser.error(str(exc))

        return commands.run_prepare(
            pdf_path=args.pdf,
            library_directory=args.library,
            name=args.name or args.pdf.stem,
            question_count=args.question_count,
        )

    if args.command == "package-list":
        try:
            validate_library_directory(args.library)
        except ValueError as exc:
            parser.error(str(exc))

        return commands.run_package_list(args.library)

    if args.command == "progress":
        try:
            validate_library_directory(args.library)
        except ValueError as exc:
            parser.error(str(exc))

        try:
            return commands.run_progress(
                library_directory=args.library,
                package_name=args.package,
            )
        except (
            LearningPackageNotFoundError,
            LearningPackageNotReadyError,
        ) as exc:
            parser.error(str(exc))

    if args.command in {"search", "ask"}:
        try:
            validate_existing_index_directory(args.index_dir)
        except ValueError as exc:
            parser.error(str(exc))

        if args.command == "search":
            return commands.run_search(
                index_directory=args.index_dir,
                query=args.query,
                limit=args.limit,
            )

        return commands.run_ask(
            index_directory=args.index_dir,
            question=args.question,
            limit=args.limit,
        )

    if args.command == "summarize":
        try:
            validate_existing_index_directory(args.index_dir)
        except ValueError as exc:
            parser.error(str(exc))

        try:
            return commands.run_summarize(
                index_directory=args.index_dir,
                document_id=args.document_id,
                max_map_new_tokens=args.max_map_new_tokens,
                max_reduce_new_tokens=args.max_reduce_new_tokens,
                max_batch_chars=args.max_batch_chars,
                force=args.force,
            )
        except DocumentNotFoundError as exc:
            parser.error(str(exc))

    if args.command == "list":
        try:
            validate_library_directory(args.index_dir)
        except ValueError as exc:
            parser.error(str(exc))

        return commands.run_list(args.index_dir)

    if args.command in {"summary-list", "summary-show"}:
        try:
            validate_library_directory(args.index_dir)
        except ValueError as exc:
            parser.error(str(exc))

        try:
            if args.command == "summary-list":
                return commands.run_summary_list(
                    index_directory=args.index_dir,
                    document_id=args.document_id,
                )

            return commands.run_summary_show(
                index_directory=args.index_dir,
                document_id=args.document_id,
                identity_fingerprint=args.identity_fingerprint,
            )
        except (
            DocumentNotFoundError,
            DocumentSummaryNotFoundError,
        ) as exc:
            parser.error(str(exc))

    if args.command in {"question-list", "question-show"}:
        try:
            validate_library_directory(args.index_dir)
        except ValueError as exc:
            parser.error(str(exc))

        try:
            if args.command == "question-list":
                return commands.run_question_list(
                    index_directory=args.index_dir,
                    document_id=args.document_id,
                )

            return commands.run_question_show(
                index_directory=args.index_dir,
                document_id=args.document_id,
                identity_fingerprint=args.identity_fingerprint,
            )
        except (
            DocumentNotFoundError,
            QuestionBankNotFoundError,
        ) as exc:
            parser.error(str(exc))

    if args.command == "question-generate":
        try:
            validate_library_directory(args.index_dir)
        except ValueError as exc:
            parser.error(str(exc))

        try:
            return commands.run_question_generate(
                index_directory=args.index_dir,
                document_id=args.document_id,
                summary_identity_fingerprint=(args.summary_identity_fingerprint),
                question_count=args.count,
                max_new_tokens=args.max_new_tokens,
                force=args.force,
            )
        except (
            DocumentNotFoundError,
            DocumentSummaryNotFoundError,
        ) as exc:
            parser.error(str(exc))

    if args.command == "study":
        package_arguments_present = args.library is not None or args.package is not None
        technical_arguments_present = (
            args.index_dir is not None
            or args.document_id is not None
            or args.question_bank_identity_fingerprint is not None
        )

        if package_arguments_present and technical_arguments_present:
            parser.error("Package and technical study arguments must not be mixed")

        if package_arguments_present:
            if args.library is None or args.package is None:
                parser.error("--library and --package must be used together")

            try:
                validate_library_directory(args.library)
            except ValueError as exc:
                parser.error(str(exc))

            try:
                return commands.run_package_study(
                    library_directory=args.library,
                    package_name=args.package,
                )
            except (
                LearningPackageNotFoundError,
                LearningPackageNotReadyError,
                StudyQuestionNotFoundError,
            ) as exc:
                parser.error(str(exc))

        if (
            args.index_dir is None
            or args.document_id is None
            or args.question_bank_identity_fingerprint is None
        ):
            parser.error(
                "study requires either --library and --package "
                "or all technical positional arguments"
            )

        try:
            validate_library_directory(args.index_dir)
        except ValueError as exc:
            parser.error(str(exc))

        try:
            return commands.run_study(
                index_directory=args.index_dir,
                document_id=args.document_id,
                question_bank_identity_fingerprint=(args.question_bank_identity_fingerprint),
            )
        except (
            DocumentNotFoundError,
            QuestionBankNotFoundError,
            StudyQuestionNotFoundError,
        ) as exc:
            parser.error(str(exc))

    if args.command in {"review-due", "review-record"}:
        try:
            validate_library_directory(args.index_dir)
        except ValueError as exc:
            parser.error(str(exc))

        try:
            if args.command == "review-due":
                return commands.run_review_due(
                    index_directory=args.index_dir,
                    document_id=args.document_id,
                    question_bank_identity_fingerprint=(args.question_bank_identity_fingerprint),
                    limit=args.limit,
                )

            return commands.run_review_record(
                index_directory=args.index_dir,
                document_id=args.document_id,
                question_bank_identity_fingerprint=(args.question_bank_identity_fingerprint),
                question_number=args.question_number,
                rating=args.rating,
            )
        except (
            DocumentNotFoundError,
            QuestionBankNotFoundError,
            StudyQuestionNotFoundError,
        ) as exc:
            parser.error(str(exc))

    if args.command == "remove":
        try:
            validate_library_directory(args.index_dir)
        except ValueError as exc:
            parser.error(str(exc))

        chunker = TextChunker(
            max_chars=DEFAULT_MAX_CHARS,
            overlap_chars=DEFAULT_OVERLAP_CHARS,
        )
        try:
            return commands.run_remove(
                document_id=args.document_id,
                chunker=chunker,
                index_directory=args.index_dir,
            )
        except DocumentNotFoundError as exc:
            parser.error(str(exc))

    if args.command == "replace":
        try:
            validate_existing_index_directory(args.index_dir)
        except ValueError as exc:
            parser.error(str(exc))

    if args.command == "index":
        try:
            validate_index_directory(args.index_dir)
        except ValueError as exc:
            parser.error(str(exc))

    try:
        chunker = TextChunker(
            max_chars=args.max_chars,
            overlap_chars=args.overlap_chars,
        )
    except ValueError as exc:
        parser.error(str(exc))

    if args.command == "replace":
        try:
            return commands.run_replace(
                document_id=args.document_id,
                pdf_path=args.pdf,
                chunker=chunker,
                index_directory=args.index_dir,
            )
        except (
            DocumentNotFoundError,
            DuplicateDocumentError,
        ) as exc:
            parser.error(str(exc))

    if args.command == "index":
        return commands.run_index(
            pdf_paths=args.pdfs,
            chunker=chunker,
            index_directory=args.index_dir,
        )

    document = PdfExtractor().extract(args.pdf)
    return commands.run_extract(document, chunker)
