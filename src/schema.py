"""
Input/output schema definitions for the civic service routing agent.

Think of this like the 'blueprint' of a house — before builders start,
everyone agrees on exactly what rooms (fields) the house will have.
Our agent must always return data in this exact shape.
"""

from typing import TypedDict

# ── Output schema ──────────────────────────────────────────────────────────────

VALID_JURISDICTION_LEVELS = {"City", "Region", "Province", "Federal", "Mixed", "Unclear"}

RESPONSE_SCHEMA: dict = {
    "service_name": "string — canonical service name from the catalog",
    "jurisdiction_level": "City | Region | Province | Federal | Mixed | Unclear",
    "responsible_body": "string — name of the government body",
    "confidence": "float in [0.0, 1.0]",
    "reasoning_summary": "string — short grounded explanation (1-3 sentences)",
    "next_steps": ["step 1 (string)", "step 2 (string)"],
    "sources": ["url_1 (string)", "url_2 (string)"],
}

# JSON Schema used for prompt enforcement
RESPONSE_JSON_SCHEMA = {
    "type": "object",
    "required": [
        "service_name",
        "jurisdiction_level",
        "responsible_body",
        "confidence",
        "reasoning_summary",
        "next_steps",
        "sources",
    ],
    "properties": {
        "service_name": {"type": "string"},
        "jurisdiction_level": {
            "type": "string",
            "enum": list(VALID_JURISDICTION_LEVELS),
        },
        "responsible_body": {"type": "string"},
        "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
        "reasoning_summary": {"type": "string"},
        "next_steps": {"type": "array", "items": {"type": "string"}, "minItems": 1},
        "sources": {"type": "array", "items": {"type": "string"}},
    },
    "additionalProperties": False,
}


# ── TypedDicts ─────────────────────────────────────────────────────────────────

class AgentInput(TypedDict):
    """What goes INTO the agent — the resident's question."""
    question: str


class ServiceResponse(TypedDict):
    """What comes OUT of the agent — the structured answer."""
    service_name: str
    jurisdiction_level: str
    responsible_body: str
    confidence: float
    reasoning_summary: str
    next_steps: list[str]
    sources: list[str]


class EvaluationRecord(TypedDict):
    """One row in the evaluation results table."""
    question: str
    expected_service_name: str
    expected_jurisdiction_level: str
    expected_responsible_body: str
    predicted_service_name: str
    predicted_jurisdiction_level: str
    predicted_responsible_body: str
    sources: list[str]
    jurisdiction_correct: bool
    body_correct: bool
    format_compliant: bool
    reasoning_quality: float


# ── Validation helper ──────────────────────────────────────────────────────────

def validate_response(response: dict) -> tuple[bool, list[str]]:
    """
    Check whether a response dict matches the expected schema.

    Returns (is_valid, list_of_errors).
    Think of it like a spell-checker — it tells you exactly what is wrong.
    """
    errors: list[str] = []
    required_keys = list(RESPONSE_JSON_SCHEMA["required"])

    for key in required_keys:
        if key not in response:
            errors.append(f"Missing required field: '{key}'")

    if "jurisdiction_level" in response:
        jl = response["jurisdiction_level"]
        if jl not in VALID_JURISDICTION_LEVELS:
            errors.append(
                f"Invalid jurisdiction_level '{jl}'. "
                f"Must be one of: {VALID_JURISDICTION_LEVELS}"
            )

    if "confidence" in response:
        conf = response["confidence"]
        if not isinstance(conf, (int, float)) or not (0.0 <= conf <= 1.0):
            errors.append(f"confidence must be a float in [0, 1], got: {conf!r}")

    if "next_steps" in response:
        if not isinstance(response["next_steps"], list) or len(response["next_steps"]) == 0:
            errors.append("next_steps must be a non-empty list")

    if "sources" in response:
        if not isinstance(response["sources"], list):
            errors.append("sources must be a list")

    return len(errors) == 0, errors
