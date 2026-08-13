from rag_learning_assistant.evaluation.models import (
    ExpectedCitation,
    GroundedEvaluationCase,
    GroundedEvaluationReport,
    GroundedEvaluationResult,
)
from rag_learning_assistant.generation import GroundedAnswer


class GroundedGenerationEvaluator:
    """Evaluate generated evidence against stable source-page expectations."""

    def evaluate(
        self,
        case: GroundedEvaluationCase,
        answer: GroundedAnswer,
    ) -> GroundedEvaluationResult:
        if answer.question != case.question:
            raise ValueError("Generated answer question does not match evaluation case")
        actual_citations = {
            ExpectedCitation(
                source=citation.source,
                page_number=citation.page_number,
            )
            for citation in answer.citations
        }

        # Preserve reference-case order so reports remain stable and readable.
        matched_citations = tuple(
            citation for citation in case.expected_citations if citation in actual_citations
        )
        missing_citations = tuple(
            citation for citation in case.expected_citations if citation not in actual_citations
        )
        expected_citations = set(case.expected_citations)
        # Sets make exact source-page comparison explicit; sorting keeps JSON
        # reports deterministic even though set iteration order is unspecified.
        unexpected_citations = tuple(
            sorted(
                actual_citations - expected_citations,
                key=lambda citation: (
                    citation.source,
                    citation.page_number,
                ),
            )
        )
        normalized_answer = answer.text.casefold()

        matched_concepts = tuple(
            concept.name
            for concept in case.expected_concepts
            if any(
                phrase.strip().casefold() in normalized_answer
                for phrase in concept.accepted_phrases
            )
        )
        missing_concepts = tuple(
            concept.name
            for concept in case.expected_concepts
            if concept.name not in matched_concepts
        )

        return GroundedEvaluationResult(
            case_id=case.case_id,
            answer_text=answer.text,
            passed=(not missing_citations and not unexpected_citations and not missing_concepts),
            matched_citations=matched_citations,
            missing_citations=missing_citations,
            unexpected_citations=unexpected_citations,
            matched_concepts=matched_concepts,
            missing_concepts=missing_concepts,
        )

    def evaluate_all(
        self,
        cases: tuple[GroundedEvaluationCase, ...],
        answers: tuple[GroundedAnswer, ...],
    ) -> GroundedEvaluationReport:
        """Evaluate ordered case-answer pairs as one reproducible report."""

        if len(cases) != len(answers):
            raise ValueError("Evaluation requires one answer for every case")

        results = tuple(
            self.evaluate(case, answer) for case, answer in zip(cases, answers, strict=True)
        )

        return GroundedEvaluationReport(results=results)
