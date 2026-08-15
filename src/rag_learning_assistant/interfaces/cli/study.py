"""Interactive terminal workflow for answering study questions."""

from collections.abc import Callable

from rag_learning_assistant.learning import (
    ReviewRating,
    StudyQuestion,
)


def conduct_study_question(
    question: StudyQuestion,
    *,
    read_line: Callable[[str], str],
    write_line: Callable[[str], None],
) -> tuple[str, ReviewRating]:
    """Capture one learner answer and its self-assigned review rating."""

    write_line(f"Question {question.number}: {question.text}")
    while True:
        answer_text = read_line("Your answer: ")
        if answer_text.strip():
            break

        write_line("Answer must not be blank.")

    # The expected answer remains hidden until the learner has committed to an
    # answer, preventing accidental recognition from replacing active recall.
    write_line(f"Expected answer: {question.expected_answer}")
    for citation in question.citations:
        write_line(
            f"Source {citation.number}: "
            f"{citation.source}, "
            f"page {citation.page_number}, "
            f"chunk {citation.chunk_index}"
        )

    while True:
        rating_text = read_line("Rating [again/hard/good/easy]: ").strip().lower()

        try:
            rating = ReviewRating(rating_text)
        except ValueError:
            write_line("Rating must be again, hard, good, or easy.")
            continue

        break
    return answer_text, rating
