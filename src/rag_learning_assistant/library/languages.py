"""Deterministic language recognition for supported PDF text."""

import re
from enum import StrEnum


class DocumentLanguage(StrEnum):
    """Languages recognized for source documents."""

    GERMAN = "de"
    ENGLISH = "en"
    UNKNOWN = "und"


_GERMAN_MARKERS = frozenset(
    {
        "aber",
        "auch",
        "auf",
        "das",
        "der",
        "die",
        "eine",
        "für",
        "ist",
        "mit",
        "nicht",
        "oder",
        "sich",
        "und",
        "von",
        "werden",
        "zu",
    }
)
_ENGLISH_MARKERS = frozenset(
    {
        "also",
        "and",
        "are",
        "as",
        "for",
        "from",
        "is",
        "not",
        "of",
        "on",
        "or",
        "that",
        "the",
        "this",
        "to",
        "with",
    }
)
_WORD_PATTERN = re.compile(r"[^\W\d_]+", re.UNICODE)


def detect_document_language(text: str) -> DocumentLanguage:
    """Recognize German or English from bounded extracted text."""

    sample = text[:50_000].casefold()
    if not sample.strip():
        return DocumentLanguage.UNKNOWN

    words = _WORD_PATTERN.findall(sample)
    german_score = sum(word in _GERMAN_MARKERS for word in words)
    english_score = sum(word in _ENGLISH_MARKERS for word in words)
    # Count distinct German characters so one umlaut-heavy term cannot dominate
    # the repeated function-word evidence from the whole sample.
    german_score += sum(character in sample for character in "äöüß") * 2

    if german_score == english_score:
        return DocumentLanguage.UNKNOWN
    if german_score > english_score:
        return DocumentLanguage.GERMAN
    return DocumentLanguage.ENGLISH
