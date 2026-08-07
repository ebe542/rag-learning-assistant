"""Parsing of structured, untrusted language-model responses."""

import json

from rag_learning_assistant.generation.models import (
    GenerationResult,
)


def parse_generation_response(raw_response: str) -> GenerationResult:
    """Parse a JSON model response into a validated result."""

    try:
        payload = json.loads(raw_response)
    except json.JSONDecodeError as exc:
        raise ValueError("Model response must be valid JSON") from exc

    if not isinstance(payload, dict):
        raise ValueError("Model response must be a JSON object")

    expected_fields = {
        "text",
        "citation_numbers",
    }

    if set(payload) != expected_fields:
        raise ValueError("Model response must contain exactly 'text' and 'citation_numbers'")

    text = payload["text"]
    citation_numbers = payload["citation_numbers"]

    if not isinstance(text, str):
        raise ValueError("Model response text must be a string")

    if not isinstance(citation_numbers, list):
        raise ValueError("Model citation_numbers must be an array")

    # Use exact type checks because bool is a subclass of int in Python,
    # while JSON true and false are never meaningful citation numbers.
    if any(type(number) is not int for number in citation_numbers):
        raise ValueError("Model citation_numbers must contain integers")

    return GenerationResult(
        text=text,
        citation_numbers=tuple(citation_numbers),
    )
