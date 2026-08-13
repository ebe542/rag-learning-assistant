import pytest

from rag_learning_assistant.evaluation import (
    ExpectedCitation,
    ExpectedConcept,
    GroundedEvaluationCase,
    GroundedEvaluationReport,
    GroundedEvaluationResult,
)


def test_grounded_evaluation_case_preserves_expected_evidence() -> None:
    expected_citation = ExpectedCitation(
        source="summarization-document.pdf",
        page_number=1,
    )

    case = GroundedEvaluationCase(
        case_id="rag-purpose",
        question="What is the purpose of the RAG Learning Assistant?",
        expected_citations=(expected_citation,),
    )

    assert case.case_id == "rag-purpose"
    assert case.question == "What is the purpose of the RAG Learning Assistant?"
    assert case.expected_citations == (expected_citation,)


@pytest.mark.parametrize("case_id", ["", "   "])
def test_grounded_evaluation_case_rejects_blank_case_id(
    case_id: str,
) -> None:
    with pytest.raises(
        ValueError,
        match="Evaluation case case_id must not be blank",
    ):
        GroundedEvaluationCase(
            case_id=case_id,
            question="What is the project purpose?",
            expected_citations=(
                ExpectedCitation(
                    source="summarization-document.pdf",
                    page_number=1,
                ),
            ),
        )


@pytest.mark.parametrize("question", ["", "   "])
def test_grounded_evaluation_case_rejects_blank_question(
    question: str,
) -> None:
    with pytest.raises(
        ValueError,
        match="Evaluation case question must not be blank",
    ):
        GroundedEvaluationCase(
            case_id="rag-purpose",
            question=question,
            expected_citations=(
                ExpectedCitation(
                    source="summarization-document.pdf",
                    page_number=1,
                ),
            ),
        )


def test_grounded_evaluation_case_requires_expected_citation() -> None:
    with pytest.raises(
        ValueError,
        match="Evaluation case requires at least one expected citation",
    ):
        GroundedEvaluationCase(
            case_id="rag-purpose",
            question="What is the project purpose?",
            expected_citations=(),
        )


@pytest.mark.parametrize("source", ["", "   "])
def test_expected_citation_rejects_blank_source(source: str) -> None:
    with pytest.raises(ValueError, match="Expected citation source must not be blank"):
        ExpectedCitation(
            source=source,
            page_number=1,
        )


@pytest.mark.parametrize("page_number", [0, -1])
def test_expected_citation_requires_positive_page_number(
    page_number: int,
) -> None:
    with pytest.raises(
        ValueError,
        match="Expected citation page_number must be positive",
    ):
        ExpectedCitation(
            source="summarization-document.pdf",
            page_number=page_number,
        )


def test_grounded_evaluation_case_rejects_duplicate_expected_citations() -> None:
    citation = ExpectedCitation(
        source="summarization-document.pdf",
        page_number=1,
    )

    with pytest.raises(
        ValueError,
        match="Evaluation case expected citations must be unique",
    ):
        GroundedEvaluationCase(
            case_id="rag-purpose",
            question="What is the project purpose?",
            expected_citations=(citation, citation),
        )


def test_evaluation_result_calculates_citation_scores() -> None:
    matched = ExpectedCitation(source="document.pdf", page_number=1)
    missing = ExpectedCitation(source="document.pdf", page_number=2)
    unexpected = ExpectedCitation(source="document.pdf", page_number=3)

    result = GroundedEvaluationResult(
        case_id="citation-scores",
        passed=False,
        matched_citations=(matched,),
        missing_citations=(missing,),
        unexpected_citations=(unexpected,),
    )

    assert result.citation_recall == 0.5
    assert result.citation_precision == 0.5


def test_evaluation_result_has_zero_precision_without_actual_citations() -> None:
    missing = ExpectedCitation(source="document.pdf", page_number=1)

    result = GroundedEvaluationResult(
        case_id="no-citations",
        passed=False,
        matched_citations=(),
        missing_citations=(missing,),
        unexpected_citations=(),
    )

    assert result.citation_recall == 0.0
    assert result.citation_precision == 0.0


def test_successful_evaluation_result_has_perfect_scores() -> None:
    matched = ExpectedCitation(source="document.pdf", page_number=1)

    result = GroundedEvaluationResult(
        case_id="perfect",
        passed=True,
        matched_citations=(matched,),
        missing_citations=(),
        unexpected_citations=(),
    )

    assert result.citation_recall == 1.0
    assert result.citation_precision == 1.0


@pytest.mark.parametrize(
    ("matched", "missing", "unexpected"),
    [
        (
            (),
            (ExpectedCitation(source="document.pdf", page_number=1),),
            (),
        ),
        (
            (ExpectedCitation(source="document.pdf", page_number=1),),
            (),
            (ExpectedCitation(source="document.pdf", page_number=2),),
        ),
    ],
)
def test_passed_evaluation_result_rejects_citation_failures(
    matched: tuple[ExpectedCitation, ...],
    missing: tuple[ExpectedCitation, ...],
    unexpected: tuple[ExpectedCitation, ...],
) -> None:
    with pytest.raises(
        ValueError,
        match="Passed evaluation result must not contain evaluation failures",
    ):
        GroundedEvaluationResult(
            case_id="inconsistent",
            passed=True,
            matched_citations=matched,
            missing_citations=missing,
            unexpected_citations=unexpected,
        )


def test_failed_evaluation_result_rejects_perfect_citation_coverage() -> None:
    with pytest.raises(
        ValueError,
        match="Failed evaluation result must contain an evaluation failure",
    ):
        GroundedEvaluationResult(
            case_id="inconsistent",
            passed=False,
            matched_citations=(ExpectedCitation(source="document.pdf", page_number=1),),
            missing_citations=(),
            unexpected_citations=(),
        )


def test_evaluation_report_calculates_aggregate_scores() -> None:
    citation_1 = ExpectedCitation(source="document.pdf", page_number=1)
    citation_2 = ExpectedCitation(source="document.pdf", page_number=2)

    report = GroundedEvaluationReport(
        results=(
            GroundedEvaluationResult(
                case_id="passed",
                passed=True,
                matched_citations=(citation_1,),
                missing_citations=(),
                unexpected_citations=(),
                matched_concepts=("purpose",),
                missing_concepts=(),
            ),
            GroundedEvaluationResult(
                case_id="failed",
                passed=False,
                matched_citations=(citation_1,),
                missing_citations=(citation_2,),
                unexpected_citations=(),
                matched_concepts=("storage",),
                missing_concepts=("metadata",),
            ),
        ),
    )

    assert report.case_count == 2
    assert report.passed_count == 1
    assert report.failed_count == 1
    assert report.pass_rate == 0.5
    assert report.citation_recall == 2 / 3
    assert report.citation_precision == 1.0
    assert report.concept_recall == 2 / 3


def test_evaluation_report_requires_at_least_one_result() -> None:
    with pytest.raises(
        ValueError,
        match="Evaluation report requires at least one result",
    ):
        GroundedEvaluationReport(results=())


def test_evaluation_report_rejects_duplicate_case_ids() -> None:
    citation = ExpectedCitation(source="document.pdf", page_number=1)
    result = GroundedEvaluationResult(
        case_id="duplicate",
        passed=True,
        matched_citations=(citation,),
        missing_citations=(),
        unexpected_citations=(),
    )

    with pytest.raises(
        ValueError,
        match="Evaluation report case IDs must be unique",
    ):
        GroundedEvaluationReport(results=(result, result))


def test_expected_concept_preserves_accepted_phrases() -> None:
    concept = ExpectedConcept(
        name="vector-dimension",
        accepted_phrases=(
            "384 dimensions",
            "384-dimensional",
        ),
    )

    assert concept.name == "vector-dimension"
    assert concept.accepted_phrases == (
        "384 dimensions",
        "384-dimensional",
    )


@pytest.mark.parametrize("name", ["", "   "])
def test_expected_concept_rejects_blank_name(name: str) -> None:
    with pytest.raises(
        ValueError,
        match="Expected concept name must not be blank",
    ):
        ExpectedConcept(
            name=name,
            accepted_phrases=("384 dimensions",),
        )


def test_expected_concept_requires_accepted_phrase() -> None:
    with pytest.raises(
        ValueError,
        match="Expected concept requires at least one accepted phrase",
    ):
        ExpectedConcept(
            name="vector-dimension",
            accepted_phrases=(),
        )


@pytest.mark.parametrize("phrase", ["", "   "])
def test_expected_concept_rejects_blank_accepted_phrase(
    phrase: str,
) -> None:
    with pytest.raises(
        ValueError,
        match="Expected concept phrases must not be blank",
    ):
        ExpectedConcept(
            name="vector-dimension",
            accepted_phrases=(phrase,),
        )


def test_expected_concept_rejects_duplicate_normalized_phrases() -> None:
    with pytest.raises(
        ValueError,
        match="Expected concept phrases must be unique",
    ):
        ExpectedConcept(
            name="vector-dimension",
            accepted_phrases=(
                "384 dimensions",
                "  384 DIMENSIONS  ",
            ),
        )


def test_grounded_evaluation_case_preserves_expected_concepts() -> None:
    concept = ExpectedConcept(
        name="vector-dimension",
        accepted_phrases=("384 dimensions",),
    )

    case = GroundedEvaluationCase(
        case_id="embedding-dimensions",
        question="How many dimensions do the vectors contain?",
        expected_citations=(ExpectedCitation(source="document.pdf", page_number=4),),
        expected_concepts=(concept,),
    )

    assert case.expected_concepts == (concept,)


def test_grounded_evaluation_case_rejects_duplicate_concept_names() -> None:
    with pytest.raises(
        ValueError,
        match="Evaluation case concept names must be unique",
    ):
        GroundedEvaluationCase(
            case_id="embedding-dimensions",
            question="How many dimensions do the vectors contain?",
            expected_citations=(ExpectedCitation(source="document.pdf", page_number=4),),
            expected_concepts=(
                ExpectedConcept(
                    name="vector-dimension",
                    accepted_phrases=("384 dimensions",),
                ),
                ExpectedConcept(
                    name="  VECTOR-DIMENSION  ",
                    accepted_phrases=("384-dimensional",),
                ),
            ),
        )


def test_evaluation_result_calculates_concept_recall() -> None:
    result = GroundedEvaluationResult(
        case_id="concepts",
        answer_text="The vectors contain 384 dimensions.",
        passed=False,
        matched_citations=(ExpectedCitation(source="document.pdf", page_number=4),),
        missing_citations=(),
        unexpected_citations=(),
        matched_concepts=("vector-dimension",),
        missing_concepts=("semantic-meaning",),
    )

    assert result.concept_recall == 0.5


def test_evaluation_result_has_perfect_concept_recall_without_concepts() -> None:
    result = GroundedEvaluationResult(
        case_id="citation-only",
        answer_text="A citation-only answer.",
        passed=True,
        matched_citations=(ExpectedCitation(source="document.pdf", page_number=1),),
        missing_citations=(),
        unexpected_citations=(),
        matched_concepts=(),
        missing_concepts=(),
    )

    assert result.concept_recall == 1.0
