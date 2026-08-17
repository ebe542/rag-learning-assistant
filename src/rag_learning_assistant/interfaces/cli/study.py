"""Interactive terminal workflow for answering study questions."""

from collections.abc import Callable

from rag_learning_assistant.learning import StudyQuestion


def capture_study_answer(
    question: StudyQuestion,
    *,
    read_line: Callable[[str], str],
    write_line: Callable[[str], None],
) -> str:
    """Capture one non-blank written learner answer."""

    write_line(f"Question {question.number}: {question.text}")

    while True:
        answer_text = read_line("Your answer: ")
        if answer_text.strip():
            return answer_text

        write_line("Answer must not be blank.")
