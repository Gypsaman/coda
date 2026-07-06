"""
Redundant, provider-independent structured-output validation.

Used as a local sanity check alongside metrics.py's regex-based JSON
extraction for Case C/F's structured contracts -- catches schema drift
even when the provider-native structured-output mode (llm_client.py's
response_schema) wasn't used for a given call (e.g. Case C's original
prompt, which deliberately stays prose-enforced -- see coda.md's
"Modernization methodology").
"""

import json
import re

import jsonschema


def validate_against_schema(output_text: str, schema: dict) -> float:
    """Extract the first JSON object from output_text and validate it against schema.

    Returns 1.0 if a JSON object is found and validates cleanly, 0.0 otherwise.
    """
    for match in re.finditer(r"\{.*\}", output_text, re.DOTALL):
        try:
            candidate = json.loads(match.group())
        except json.JSONDecodeError:
            continue
        try:
            jsonschema.validate(candidate, schema)
            return 1.0
        except jsonschema.ValidationError:
            continue
    return 0.0
