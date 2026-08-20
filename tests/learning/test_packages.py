from uuid import UUID

import pytest

from rag_learning_assistant.learning import (
    LearningPackage,
    LearningPackageStatus,
)


def test_learning_package_tracks_active_learning_material() -> None:
    package = LearningPackage(
        id=UUID("aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa"),
        name="python-basics",
        document_id=UUID("bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb"),
        status=LearningPackageStatus.READY,
        summary_identity_fingerprint="c" * 64,
        question_bank_identity_fingerprint="d" * 64,
    )

    assert package.name == "python-basics"
    assert package.status is LearningPackageStatus.READY
    assert package.summary_identity_fingerprint == "c" * 64
    assert package.question_bank_identity_fingerprint == "d" * 64


@pytest.mark.parametrize("name", ["", "   "])
def test_learning_package_rejects_blank_name(name: str) -> None:
    with pytest.raises(ValueError, match="Learning package name must not be blank"):
        LearningPackage(
            id=UUID("aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa"),
            name=name,
            document_id=UUID("bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb"),
            status=LearningPackageStatus.INDEXED,
        )


def test_summarized_package_requires_summary_identity() -> None:
    with pytest.raises(
        ValueError,
        match="Summarized learning package requires a summary identity",
    ):
        LearningPackage(
            id=UUID("aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa"),
            name="python-basics",
            document_id=UUID("bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb"),
            status=LearningPackageStatus.SUMMARIZED,
        )


def test_ready_package_requires_question_bank_identity() -> None:
    with pytest.raises(
        ValueError,
        match="Ready learning package requires a question-bank identity",
    ):
        LearningPackage(
            id=UUID("aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa"),
            name="python-basics",
            document_id=UUID("bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb"),
            status=LearningPackageStatus.READY,
            summary_identity_fingerprint="c" * 64,
        )


def test_indexed_package_rejects_derived_material() -> None:
    with pytest.raises(
        ValueError,
        match="Indexed learning package must not reference derived material",
    ):
        LearningPackage(
            id=UUID("aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa"),
            name="python-basics",
            document_id=UUID("bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb"),
            status=LearningPackageStatus.INDEXED,
            summary_identity_fingerprint="c" * 64,
        )


def test_summarized_package_rejects_question_bank_identity() -> None:
    with pytest.raises(
        ValueError,
        match="Summarized learning package must not reference a question bank",
    ):
        LearningPackage(
            id=UUID("aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa"),
            name="python-basics",
            document_id=UUID("bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb"),
            status=LearningPackageStatus.SUMMARIZED,
            summary_identity_fingerprint="c" * 64,
            question_bank_identity_fingerprint="d" * 64,
        )


@pytest.mark.parametrize(
    "fingerprint",
    [
        "",
        "a" * 63,
        "a" * 65,
        "G" * 64,
        "A" * 64,
    ],
)
def test_learning_package_rejects_invalid_summary_identity(
    fingerprint: str,
) -> None:
    with pytest.raises(
        ValueError,
        match="Summary identity must be a lowercase SHA-256 fingerprint",
    ):
        LearningPackage(
            id=UUID("aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa"),
            name="python-basics",
            document_id=UUID("bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb"),
            status=LearningPackageStatus.SUMMARIZED,
            summary_identity_fingerprint=fingerprint,
        )


@pytest.mark.parametrize(
    "fingerprint",
    [
        "",
        "a" * 63,
        "a" * 65,
        "G" * 64,
        "A" * 64,
    ],
)
def test_learning_package_rejects_invalid_question_bank_identity(
    fingerprint: str,
) -> None:
    with pytest.raises(
        ValueError,
        match="Question-bank identity must be a lowercase SHA-256 fingerprint",
    ):
        LearningPackage(
            id=UUID("aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa"),
            name="python-basics",
            document_id=UUID("bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb"),
            status=LearningPackageStatus.READY,
            summary_identity_fingerprint="c" * 64,
            question_bank_identity_fingerprint=fingerprint,
        )
