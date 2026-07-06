"""Shared helpers used across CODA's LLM-as-classifier and LLM-as-judge evaluators."""


def strip_json_fences(text: str) -> str:
    """Strip a leading/trailing markdown code fence (```json ... ```) from an LLM response."""
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.split("\n", 1)[1] if "\n" in cleaned else cleaned[3:]
    if cleaned.endswith("```"):
        cleaned = cleaned[:-3]
    cleaned = cleaned.strip()
    if cleaned.startswith("json"):
        cleaned = cleaned[4:].strip()
    return cleaned
