from builtins import ValueError

import pytest

from rag_learning_assistant.evaluation import (
    ExpectedCitation,
    ExpectedConcept,
    GroundedEvaluationCase,
    GroundedGenerationEvaluator,
)
from rag_learning_assistant.generation import Citation, GroundedAnswer


def test_evaluator_accepts_answer_with_all_expected_citations() -> None:
    case = GroundedEvaluationCase(
        case_id="rag-purpose",
        question="What is the purpose of the RAG Learning Assistant?",
        expected_citations=(
            ExpectedCitation(
                source="summarization-document.pdf",
                page_number=1,
            ),
        ),
    )
    answer = GroundedAnswer(
        question=case.question,
        text="The assistant provides source-grounded learning support.",
        citations=(
            Citation(
                number=1,
                source="summarization-document.pdf",
                page_number=1,
                chunk_index=0,
                excerpt="The RAG Learning Assistant is a source-grounded study system.",
            ),
        ),
    )

    result = GroundedGenerationEvaluator().evaluate(case, answer)

    assert result.case_id == "rag-purpose"
    assert result.passed is True
    assert result.matched_citations == case.expected_citations
    assert result.missing_citations == ()


def test_evaluator_reports_missing_expected_citation() -> None:
    expected_citation = ExpectedCitation(
        source="summarization-document.pdf",
        page_number=3,
    )
    case = GroundedEvaluationCase(
        case_id="extraction",
        question="What does extraction create?",
        expected_citations=(expected_citation,),
    )
    answer = GroundedAnswer(
        question=case.question,
        text="Extraction creates immutable page models.",
        citations=(
            Citation(
                number=1,
                source="summarization-document.pdf",
                page_number=2,
                chunk_index=2,
                excerpt="An unrelated passage.",
            ),
        ),
    )

    result = GroundedGenerationEvaluator().evaluate(case, answer)

    assert result.passed is False
    assert result.matched_citations == ()
    assert result.missing_citations == (expected_citation,)


def test_evaluator_rejects_answer_for_different_question() -> None:
    case = GroundedEvaluationCase(
        case_id="rag-purpose",
        question="What is the project purpose?",
        expected_citations=(
            ExpectedCitation(
                source="summarization-document.pdf",
                page_number=1,
            ),
        ),
    )
    answer = GroundedAnswer(
        question="How does caching work?",
        text="The cache stores completed map results.",
        citations=(
            Citation(
                number=1,
                source="summarization-document.pdf",
                page_number=9,
                chunk_index=16,
                excerpt="The summary cache stores each validated map result.",
            ),
        ),
    )

    with pytest.raises(
        ValueError,
        match="Generated answer question does not match evaluation case",
    ):
        GroundedGenerationEvaluator().evaluate(case, answer)


def test_evaluator_reports_unexpected_citations() -> None:
    expected_citation = ExpectedCitation(
        source="summarization-document.pdf",
        page_number=1,
    )
    unexpected_citation = ExpectedCitation(
        source="summarization-document.pdf",
        page_number=8,
    )
    case = GroundedEvaluationCase(
        case_id="rag-purpose",
        question="What is the project purpose?",
        expected_citations=(expected_citation,),
    )
    answer = GroundedAnswer(
        question=case.question,
        text="The project provides source-grounded learning support.",
        citations=(
            Citation(
                number=1,
                source=expected_citation.source,
                page_number=expected_citation.page_number,
                chunk_index=0,
                excerpt="The RAG Learning Assistant is a source-grounded study system.",
            ),
            Citation(
                number=2,
                source=unexpected_citation.source,
                page_number=unexpected_citation.page_number,
                chunk_index=14,
                excerpt="Operational concerns unrelated to the project purpose.",
            ),
        ),
    )

    result = GroundedGenerationEvaluator().evaluate(case, answer)

    assert result.passed is False
    assert result.matched_citations == (expected_citation,)
    assert result.missing_citations == ()
    assert result.unexpected_citations == (unexpected_citation,)


def test_evaluator_builds_report_for_multiple_cases() -> None:
    first_case = GroundedEvaluationCase(
        case_id="purpose",
        question="What is the project purpose?",
        expected_citations=(
            ExpectedCitation(
                source="summarization-document.pdf",
                page_number=1,
            ),
        ),
    )
    second_case = GroundedEvaluationCase(
        case_id="storage",
        question="Which persistent stores are used?",
        expected_citations=(
            ExpectedCitation(
                source="summarization-document.pdf",
                page_number=5,
            ),
        ),
    )
    answers = (
        GroundedAnswer(
            question=first_case.question,
            text="The project provides source-grounded learning support.",
            citations=(
                Citation(
                    number=1,
                    source="summarization-document.pdf",
                    page_number=1,
                    chunk_index=0,
                    excerpt="A source-grounded study system.",
                ),
            ),
        ),
        GroundedAnswer(
            question=second_case.question,
            text="The project uses FAISS and SQLite.",
            citations=(
                Citation(
                    number=1,
                    source="summarization-document.pdf",
                    page_number=5,
                    chunk_index=8,
                    excerpt="Persistent storage uses FAISS and SQLite.",
                ),
            ),
        ),
    )

    report = GroundedGenerationEvaluator().evaluate_all(
        cases=(first_case, second_case),
        answers=answers,
    )

    assert report.case_count == 2
    assert report.passed_count == 2
    assert report.failed_count == 0
    assert report.pass_rate == 1.0


def test_evaluator_rejects_different_case_and_answer_counts() -> None:
    case = GroundedEvaluationCase(
        case_id="purpose",
        question="What is the project purpose?",
        expected_citations=(
            ExpectedCitation(
                source="summarization-document.pdf",
                page_number=1,
            ),
        ),
    )

    with pytest.raises(
        ValueError,
        match="Evaluation requires one answer for every case",
    ):
        GroundedGenerationEvaluator().evaluate_all(
            cases=(case,),
            answers=(),
        )


def test_evaluator_matches_concept_case_insensitively() -> None:
    case = GroundedEvaluationCase(
        case_id="embedding-dimensions",
        question="How many dimensions do the vectors contain?",
        expected_citations=(ExpectedCitation(source="document.pdf", page_number=4),),
        expected_concepts=(
            ExpectedConcept(
                name="vector-dimension",
                accepted_phrases=(
                    "384 dimensions",
                    "384-dimensional",
                ),
            ),
        ),
    )
    answer = GroundedAnswer(
        question=case.question,
        text="The vectors contain 384 DIMENSIONS.",
        citations=(
            Citation(
                number=1,
                source="document.pdf",
                page_number=4,
                chunk_index=6,
                excerpt="Vectors contain 384 dimensions.",
            ),
        ),
    )

    result = GroundedGenerationEvaluator().evaluate(case, answer)

    assert result.passed is True
    assert result.answer_text == answer.text
    assert result.matched_concepts == ("vector-dimension",)
    assert result.missing_concepts == ()
    assert result.concept_recall == 1.0


def test_evaluator_fails_when_expected_concept_is_missing() -> None:
    case = GroundedEvaluationCase(
        case_id="embedding-dimensions",
        question="How many dimensions do the vectors contain?",
        expected_citations=(ExpectedCitation(source="document.pdf", page_number=4),),
        expected_concepts=(
            ExpectedConcept(
                name="vector-dimension",
                accepted_phrases=("384 dimensions",),
            ),
        ),
    )
    answer = GroundedAnswer(
        question=case.question,
        text="The project uses a multilingual embedding model.",
        citations=(
            Citation(
                number=1,
                source="document.pdf",
                page_number=4,
                chunk_index=6,
                excerpt="Vectors contain 384 dimensions.",
            ),
        ),
    )

    result = GroundedGenerationEvaluator().evaluate(case, answer)

    assert result.passed is False
    assert result.matched_concepts == ()
    assert result.missing_concepts == ("vector-dimension",)
    assert result.concept_recall == 0.0


def test_evaluator_accepts_untrusted_context_wording_for_retrieved_evidence() -> None:
    case = GroundedEvaluationCase(
        case_id="grounded-generation",
        question="How does the application keep generated answers grounded?",
        expected_citations=(
            ExpectedCitation(
                source="summarization-document.pdf",
                page_number=7,
            ),
        ),
        expected_concepts=(
            ExpectedConcept(
                name="retrieved-evidence",
                accepted_phrases=(
                    "retrieved evidence",
                    "retrieved contexts",
                    "retrieved source passages",
                    "contexts as untrusted data",
                ),
            ),
        ),
    )
    answer = GroundedAnswer(
        question=case.question,
        text=(
            "The prompt forbids unsupported prior knowledge and marks contexts as untrusted data."
        ),
        citations=(
            Citation(
                number=1,
                source="summarization-document.pdf",
                page_number=7,
                chunk_index=12,
                excerpt=(
                    "The application marks contexts as untrusted data and requests citations."
                ),
            ),
        ),
    )

    result = GroundedGenerationEvaluator().evaluate(case, answer)

    assert result.passed is True
    assert result.matched_concepts == ("retrieved-evidence",)
    assert result.missing_concepts == ()
