"""Evaluate grounded generation against versioned reference cases."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Callable, Sequence
from contextlib import suppress
from pathlib import Path
from typing import Any, Protocol

from dotenv import load_dotenv

from rag_learning_assistant.evaluation import (
    ExpectedCitation,
    ExpectedConcept,
    GroundedEvaluationCase,
    GroundedEvaluationReport,
    GroundedGenerationEvaluator,
)
from rag_learning_assistant.generation import GroundedAnswer
from rag_learning_assistant.interfaces.cli.commands import (
    build_question_answering_service,
)
from rag_learning_assistant.interfaces.cli.parser import (
    DEFAULT_RESULT_LIMIT,
    positive_int,
    validate_existing_index_directory,
)

SCHEMA_VERSION = 1
REPORT_SCHEMA_VERSION = 1
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CASES_PATH = PROJECT_ROOT / "benchmarks" / "evaluation" / "grounded-generation-cases.json"


class QuestionAnswerer(Protocol):
    """Answer evaluation questions through the production QA boundary."""

    def answer(self, question: str, limit: int) -> GroundedAnswer: ...


def build_parser() -> argparse.ArgumentParser:
    """Build arguments for the grounded-generation evaluation."""

    parser = argparse.ArgumentParser(
        description=("Evaluate grounded answers against versioned citation expectations"),
    )
    parser.add_argument(
        "index_dir",
        type=Path,
        help="Persistent library index containing the benchmark document",
    )
    parser.add_argument(
        "--cases",
        type=Path,
        default=DEFAULT_CASES_PATH,
        help="Versioned evaluation cases JSON file",
    )
    parser.add_argument(
        "--limit",
        type=positive_int,
        default=DEFAULT_RESULT_LIMIT,
        help=(
            f"Maximum retrieved passages per evaluation question (default: {DEFAULT_RESULT_LIMIT})"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional path for the generated JSON report",
    )
    return parser


def load_cases(path: Path) -> tuple[GroundedEvaluationCase, ...]:
    """Load deterministic evaluation cases from a versioned JSON document."""

    try:
        payload: Any = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("Evaluation cases file has an invalid schema") from exc

    if (
        not isinstance(payload, dict)
        or payload.get("schema_version") != SCHEMA_VERSION
        or not isinstance(payload.get("cases"), list)
    ):
        raise ValueError("Evaluation cases file has an invalid schema")

    raw_cases = payload["cases"]
    if not raw_cases:
        raise ValueError("Evaluation cases file requires at least one case")

    try:
        cases = tuple(
            GroundedEvaluationCase(
                case_id=item["case_id"],
                question=item["question"],
                expected_citations=tuple(
                    ExpectedCitation(
                        source=citation["source"],
                        page_number=citation["page_number"],
                    )
                    for citation in item["expected_citations"]
                ),
                expected_concepts=tuple(
                    ExpectedConcept(
                        name=concept["name"],
                        accepted_phrases=tuple(concept["accepted_phrases"]),
                    )
                    for concept in item.get(
                        "expected_concepts",
                        [],
                    )
                ),
            )
            for item in raw_cases
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("Evaluation cases file has an invalid schema") from exc

    case_ids = [case.case_id for case in cases]
    if len(set(case_ids)) != len(case_ids):
        raise ValueError("Evaluation case IDs must be unique")

    return cases


def evaluate_cases(
    cases: tuple[GroundedEvaluationCase, ...],
    answerer: QuestionAnswerer,
    result_limit: int,
    progress: Callable[[int, int, str], None] | None = None,
) -> GroundedEvaluationReport:
    """Run ordered reference cases through one question-answering service."""

    if result_limit < 1:
        raise ValueError("Evaluation result limit must be positive")

    answers: list[GroundedAnswer] = []
    total = len(cases)

    for current, case in enumerate(cases, start=1):
        if progress is not None:
            progress(current, total, case.case_id)

        answers.append(answerer.answer(case.question, limit=result_limit))

    return GroundedGenerationEvaluator().evaluate_all(
        cases=cases,
        answers=tuple(answers),
    )


def write_progress(current: int, total: int, case_id: str) -> None:
    """Write human-readable progress without contaminating JSON stdout."""

    print(
        f"Evaluating case {current}/{total}: {case_id}...",
        file=sys.stderr,
    )


def serialize_report(report: GroundedEvaluationReport) -> dict[str, object]:
    """Convert an evaluation report into stable JSON-compatible data."""

    def serialize_citations(
        citations: tuple[ExpectedCitation, ...],
    ) -> list[dict[str, object]]:
        return [
            {
                "source": citation.source,
                "page_number": citation.page_number,
            }
            for citation in citations
        ]

    return {
        "schema_version": REPORT_SCHEMA_VERSION,
        "summary": {
            "case_count": report.case_count,
            "passed_count": report.passed_count,
            "failed_count": report.failed_count,
            "pass_rate": report.pass_rate,
            "citation_recall": report.citation_recall,
            "citation_precision": report.citation_precision,
            "concept_recall": report.concept_recall,
        },
        "results": [
            {
                "case_id": result.case_id,
                "passed": result.passed,
                "answer_text": result.answer_text,
                "citation_recall": result.citation_recall,
                "citation_precision": result.citation_precision,
                "concept_recall": result.concept_recall,
                "matched_citations": serialize_citations(result.matched_citations),
                "missing_citations": serialize_citations(result.missing_citations),
                "unexpected_citations": serialize_citations(result.unexpected_citations),
                "matched_concepts": list(result.matched_concepts),
                "missing_concepts": list(result.missing_concepts),
            }
            for result in report.results
        ],
    }


def main(argv: Sequence[str] | None = None) -> int:
    """Run grounded-generation evaluation through the production QA pipeline."""

    # Authentication improves Hugging Face downloads but remains optional for
    # already cached public models and keeps the evaluation reproducible offline.
    with suppress(Exception):
        load_dotenv(dotenv_path=PROJECT_ROOT / ".env")

    args = build_parser().parse_args(argv)

    try:
        validate_existing_index_directory(args.index_dir)

        cases = load_cases(args.cases)
        answerer = build_question_answering_service(args.index_dir)
        report = evaluate_cases(
            cases=cases,
            answerer=answerer,
            result_limit=args.limit,
            progress=write_progress,
        )
        payload = serialize_report(report)
        serialized = json.dumps(payload, ensure_ascii=False, indent=2)

        if args.output is not None:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(f"{serialized}\n", encoding="utf-8")
    except (OSError, RuntimeError, ValueError) as exc:
        print(
            f"Grounded generation evaluation failed: {exc}",
            file=sys.stderr,
        )
        return 1

    print(serialized)
    return 1 if report.failed_count else 0


if __name__ == "__main__":
    raise SystemExit(main())
