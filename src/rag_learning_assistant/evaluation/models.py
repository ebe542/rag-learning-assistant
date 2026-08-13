from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ExpectedCitation:
    """Identify stable source evidence without depending on chunk boundaries."""

    source: str
    page_number: int

    def __post_init__(self) -> None:
        if not self.source.strip():
            raise ValueError("Expected citation source must not be blank")

        if self.page_number < 1:
            raise ValueError("Expected citation page_number must be positive")


@dataclass(frozen=True, slots=True)
class ExpectedConcept:
    """Describe one required idea through accepted textual expressions."""

    name: str
    accepted_phrases: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("Expected concept name must not be blank")

        if not self.accepted_phrases:
            raise ValueError("Expected concept requires at least one accepted phrase")

        if any(not phrase.strip() for phrase in self.accepted_phrases):
            raise ValueError("Expected concept phrases must not be blank")

        normalized_phrases = {phrase.strip().casefold() for phrase in self.accepted_phrases}
        if len(normalized_phrases) != len(self.accepted_phrases):
            raise ValueError("Expected concept phrases must be unique")


@dataclass(frozen=True, slots=True)
class GroundedEvaluationCase:
    """Describe one grounded question and its expected source evidence."""

    case_id: str
    question: str
    expected_citations: tuple[ExpectedCitation, ...]
    expected_concepts: tuple[ExpectedConcept, ...] = ()

    def __post_init__(self) -> None:
        if not self.case_id.strip():
            raise ValueError("Evaluation case case_id must not be blank")

        if not self.question.strip():
            raise ValueError("Evaluation case question must not be blank")

        # An evidence-free case could only measure plausible wording, not whether
        # the generated answer is actually grounded in the benchmark document.
        if not self.expected_citations:
            raise ValueError("Evaluation case requires at least one expected citation")

        if len(set(self.expected_citations)) != len(self.expected_citations):
            raise ValueError("Evaluation case expected citations must be unique")

        normalized_concept_names = {
            concept.name.strip().casefold() for concept in self.expected_concepts
        }
        if len(normalized_concept_names) != len(self.expected_concepts):
            raise ValueError("Evaluation case concept names must be unique")


@dataclass(frozen=True, slots=True)
class GroundedEvaluationResult:
    """Record deterministic evidence coverage for one evaluation case."""

    case_id: str
    passed: bool
    matched_citations: tuple[ExpectedCitation, ...]
    missing_citations: tuple[ExpectedCitation, ...]
    unexpected_citations: tuple[ExpectedCitation, ...]
    answer_text: str = ""
    matched_concepts: tuple[str, ...] = ()
    missing_concepts: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        has_evaluation_failure = bool(
            self.missing_citations or self.unexpected_citations or self.missing_concepts
        )

        if self.passed and has_evaluation_failure:
            raise ValueError("Passed evaluation result must not contain evaluation failures")

        if not self.passed and not has_evaluation_failure:
            raise ValueError("Failed evaluation result must contain an evaluation failure")

    @property
    def concept_recall(self) -> float:
        expected_count = len(self.matched_concepts) + len(self.missing_concepts)
        if expected_count == 0:
            return 1.0
        return len(self.matched_concepts) / expected_count

    @property
    def citation_recall(self) -> float:
        expected_count = len(self.matched_citations) + len(self.missing_citations)
        if expected_count == 0:
            return 0.0
        return len(self.matched_citations) / expected_count

    @property
    def citation_precision(self) -> float:
        actual_count = len(self.matched_citations) + len(self.unexpected_citations)
        if actual_count == 0:
            return 0.0
        return len(self.matched_citations) / actual_count


@dataclass(frozen=True, slots=True)
class GroundedEvaluationReport:
    """Aggregate deterministic citation metrics across evaluation cases."""

    results: tuple[GroundedEvaluationResult, ...]

    def __post_init__(self) -> None:
        if not self.results:
            raise ValueError("Evaluation report requires at least one result")

        case_ids = [result.case_id for result in self.results]
        if len(set(case_ids)) != len(case_ids):
            raise ValueError("Evaluation report case IDs must be unique")

    @property
    def case_count(self) -> int:
        return len(self.results)

    @property
    def passed_count(self) -> int:
        return sum(result.passed for result in self.results)

    @property
    def failed_count(self) -> int:
        return self.case_count - self.passed_count

    @property
    def pass_rate(self) -> float:
        return self.passed_count / self.case_count

    @property
    def citation_recall(self) -> float:
        matched_count = sum(len(result.matched_citations) for result in self.results)
        expected_count = matched_count + sum(
            len(result.missing_citations) for result in self.results
        )
        if expected_count == 0:
            return 0.0
        return matched_count / expected_count

    @property
    def citation_precision(self) -> float:
        matched_count = sum(len(result.matched_citations) for result in self.results)
        actual_count = matched_count + sum(
            len(result.unexpected_citations) for result in self.results
        )
        if actual_count == 0:
            return 0.0
        return matched_count / actual_count

    @property
    def concept_recall(self) -> float:
        matched_count = sum(len(result.matched_concepts) for result in self.results)
        expected_count = matched_count + sum(
            len(result.missing_concepts) for result in self.results
        )
        if expected_count == 0:
            return 1.0
        return matched_count / expected_count
