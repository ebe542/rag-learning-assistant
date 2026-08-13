import json
from pathlib import Path

import pytest

from rag_learning_assistant.evaluation import (
    ExpectedCitation,
    ExpectedConcept,
    GroundedEvaluationCase,
)
from rag_learning_assistant.generation import Citation, GroundedAnswer
from scripts import evaluate_grounded_generation
from scripts.evaluate_grounded_generation import (
    DEFAULT_CASES_PATH,
    build_parser,
    evaluate_cases,
    load_cases,
    serialize_report,
)


def test_load_cases_reads_versioned_json_file(tmp_path) -> None:
    path = tmp_path / "cases.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "cases": [
                    {
                        "case_id": "rag-purpose",
                        "question": ("What is the purpose of the RAG Learning Assistant?"),
                        "expected_citations": [
                            {
                                "source": "summarization-document.pdf",
                                "page_number": 1,
                            }
                        ],
                        "expected_concepts": [
                            {
                                "name": "source-grounded-purpose",
                                "accepted_phrases": [
                                    "source-grounded study system",
                                    "source-grounded learning support",
                                ],
                            }
                        ],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    cases = load_cases(path)

    assert cases == (
        GroundedEvaluationCase(
            case_id="rag-purpose",
            question="What is the purpose of the RAG Learning Assistant?",
            expected_citations=(
                ExpectedCitation(
                    source="summarization-document.pdf",
                    page_number=1,
                ),
            ),
            expected_concepts=(
                ExpectedConcept(
                    name="source-grounded-purpose",
                    accepted_phrases=(
                        "source-grounded study system",
                        "source-grounded learning support",
                    ),
                ),
            ),
        ),
    )


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"schema_version": 2, "cases": []},
        {"schema_version": 1},
        {"schema_version": 1, "cases": "invalid"},
    ],
)
def test_load_cases_rejects_invalid_document_schema(
    tmp_path,
    payload: object,
) -> None:
    path = tmp_path / "cases.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(
        ValueError,
        match="Evaluation cases file has an invalid schema",
    ):
        load_cases(path)


def test_load_cases_rejects_empty_case_list(tmp_path) -> None:
    path = tmp_path / "cases.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "cases": [],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError,
        match="Evaluation cases file requires at least one case",
    ):
        load_cases(path)


def test_load_cases_rejects_duplicate_case_ids(tmp_path) -> None:
    case = {
        "case_id": "duplicate",
        "question": "What is the project purpose?",
        "expected_citations": [
            {
                "source": "summarization-document.pdf",
                "page_number": 1,
            }
        ],
    }
    path = tmp_path / "cases.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "cases": [case, case],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError,
        match="Evaluation case IDs must be unique",
    ):
        load_cases(path)


class RecordingAnswerer:
    def __init__(self) -> None:
        self.calls: list[tuple[str, int]] = []

    def answer(self, question: str, limit: int) -> GroundedAnswer:
        self.calls.append((question, limit))
        return GroundedAnswer(
            question=question,
            text="The project is a source-grounded study system.",
            citations=(
                Citation(
                    number=1,
                    source="summarization-document.pdf",
                    page_number=1,
                    chunk_index=0,
                    excerpt="A source-grounded study system.",
                ),
            ),
        )


def test_evaluate_cases_answers_each_case_and_builds_report() -> None:
    cases = (
        GroundedEvaluationCase(
            case_id="project-purpose",
            question="What is the purpose of the RAG Learning Assistant?",
            expected_citations=(
                ExpectedCitation(
                    source="summarization-document.pdf",
                    page_number=1,
                ),
            ),
        ),
    )
    answerer = RecordingAnswerer()

    report = evaluate_cases(
        cases=cases,
        answerer=answerer,
        result_limit=3,
    )

    assert answerer.calls == [
        ("What is the purpose of the RAG Learning Assistant?", 3),
    ]
    assert report.case_count == 1
    assert report.passed_count == 1
    assert report.citation_recall == 1.0
    assert report.citation_precision == 1.0


def test_serialize_report_contains_aggregate_and_case_metrics() -> None:
    cases = (
        GroundedEvaluationCase(
            case_id="project-purpose",
            question="What is the project purpose?",
            expected_citations=(
                ExpectedCitation(
                    source="summarization-document.pdf",
                    page_number=1,
                ),
            ),
            expected_concepts=(
                ExpectedConcept(
                    name="source-grounded-purpose",
                    accepted_phrases=(
                        "source-grounded study system",
                        "source-grounded learning support",
                    ),
                ),
            ),
        ),
    )
    report = evaluate_cases(
        cases=cases,
        answerer=RecordingAnswerer(),
        result_limit=3,
    )

    assert serialize_report(report) == {
        "schema_version": 1,
        "summary": {
            "case_count": 1,
            "passed_count": 1,
            "failed_count": 0,
            "pass_rate": 1.0,
            "citation_recall": 1.0,
            "citation_precision": 1.0,
            "concept_recall": report.concept_recall,
        },
        "results": [
            {
                "case_id": "project-purpose",
                "passed": True,
                "answer_text": "The project is a source-grounded study system.",
                "citation_recall": 1.0,
                "citation_precision": 1.0,
                "concept_recall": 1.0,
                "matched_citations": [
                    {
                        "source": "summarization-document.pdf",
                        "page_number": 1,
                    }
                ],
                "missing_citations": [],
                "unexpected_citations": [],
                "matched_concepts": ["source-grounded-purpose"],
                "missing_concepts": [],
            }
        ],
    }


def test_parser_uses_versioned_cases_and_default_result_limit() -> None:
    args = build_parser().parse_args(["local-data/indexes/summarization-benchmark"])

    assert args.index_dir == Path("local-data/indexes/summarization-benchmark")
    assert args.cases == DEFAULT_CASES_PATH
    assert args.limit == 5
    assert args.output is None


def test_parser_accepts_evaluation_options() -> None:
    args = build_parser().parse_args(
        [
            "local-data/indexes/summarization-benchmark",
            "--cases",
            "custom-cases.json",
            "--limit",
            "3",
            "--output",
            "reports/evaluation.json",
        ]
    )

    assert args.cases == Path("custom-cases.json")
    assert args.limit == 3
    assert args.output == Path("reports/evaluation.json")


def test_main_evaluates_cases_and_writes_json_report(
    tmp_path,
    monkeypatch,
    capsys,
) -> None:
    cases_path = tmp_path / "cases.json"
    cases_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "cases": [
                    {
                        "case_id": "project-purpose",
                        "question": "What is the project purpose?",
                        "expected_citations": [
                            {
                                "source": "summarization-document.pdf",
                                "page_number": 1,
                            }
                        ],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    output_path = tmp_path / "reports" / "evaluation.json"
    index_directory = tmp_path / "index"
    answerer = RecordingAnswerer()
    validated_directories: list[Path] = []

    monkeypatch.setattr(
        evaluate_grounded_generation,
        "validate_existing_index_directory",
        validated_directories.append,
    )
    monkeypatch.setattr(
        evaluate_grounded_generation,
        "build_question_answering_service",
        lambda path: answerer,
    )

    exit_code = evaluate_grounded_generation.main(
        [
            str(index_directory),
            "--cases",
            str(cases_path),
            "--limit",
            "3",
            "--output",
            str(output_path),
        ]
    )

    payload = json.loads(output_path.read_text(encoding="utf-8"))

    assert exit_code == 0
    assert validated_directories == [index_directory]
    assert answerer.calls == [("What is the project purpose?", 3)]
    assert payload["summary"]["passed_count"] == 1
    assert json.loads(capsys.readouterr().out) == payload


def test_main_reports_invalid_index_before_building_service(
    tmp_path,
    monkeypatch,
    capsys,
) -> None:
    index_directory = tmp_path / "missing-index"

    def reject_index(path: Path) -> None:
        assert path == index_directory
        raise ValueError("Index directory does not exist")

    monkeypatch.setattr(
        evaluate_grounded_generation,
        "validate_existing_index_directory",
        reject_index,
    )

    def fail_if_built(path: Path) -> None:
        raise AssertionError("QA service must not be built for an invalid index")

    monkeypatch.setattr(
        evaluate_grounded_generation,
        "build_question_answering_service",
        fail_if_built,
    )

    exit_code = evaluate_grounded_generation.main([str(index_directory)])

    captured = capsys.readouterr()

    assert exit_code == 1
    assert captured.out == ""
    assert captured.err == "Grounded generation evaluation failed: Index directory does not exist\n"


def test_evaluate_cases_reports_progress_before_each_answer() -> None:
    cases = (
        GroundedEvaluationCase(
            case_id="first",
            question="First question?",
            expected_citations=(ExpectedCitation(source="document.pdf", page_number=1),),
        ),
        GroundedEvaluationCase(
            case_id="second",
            question="Second question?",
            expected_citations=(ExpectedCitation(source="document.pdf", page_number=1),),
        ),
    )
    progress: list[tuple[int, int, str]] = []

    evaluate_cases(
        cases=cases,
        answerer=RecordingAnswerer(),
        result_limit=3,
        progress=lambda current, total, case_id: progress.append((current, total, case_id)),
    )

    assert progress == [
        (1, 2, "first"),
        (2, 2, "second"),
    ]


class MissingConceptAnswerer:
    def answer(self, question: str, limit: int) -> GroundedAnswer:
        return GroundedAnswer(
            question=question,
            text="The answer omits the expected concept.",
            citations=(
                Citation(
                    number=1,
                    source="summarization-document.pdf",
                    page_number=1,
                    chunk_index=0,
                    excerpt="A source-grounded study system.",
                ),
            ),
        )


def test_main_returns_one_after_writing_failed_evaluation_report(
    tmp_path,
    monkeypatch,
    capsys,
) -> None:
    cases_path = tmp_path / "cases.json"
    cases_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "cases": [
                    {
                        "case_id": "project-purpose",
                        "question": "What is the project purpose?",
                        "expected_citations": [
                            {
                                "source": "summarization-document.pdf",
                                "page_number": 1,
                            }
                        ],
                        "expected_concepts": [
                            {
                                "name": "source-grounded-purpose",
                                "accepted_phrases": ["source-grounded study system"],
                            }
                        ],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    output_path = tmp_path / "evaluation.json"

    monkeypatch.setattr(
        evaluate_grounded_generation,
        "validate_existing_index_directory",
        lambda path: None,
    )
    monkeypatch.setattr(
        evaluate_grounded_generation,
        "build_question_answering_service",
        lambda path: MissingConceptAnswerer(),
    )

    exit_code = evaluate_grounded_generation.main(
        [
            str(tmp_path / "index"),
            "--cases",
            str(cases_path),
            "--output",
            str(output_path),
        ]
    )

    payload = json.loads(output_path.read_text(encoding="utf-8"))

    assert exit_code == 1
    assert payload["summary"]["failed_count"] == 1
    assert payload["results"][0]["missing_concepts"] == ["source-grounded-purpose"]
    assert json.loads(capsys.readouterr().out) == payload
