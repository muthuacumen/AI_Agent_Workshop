"""
Evaluation functions for the civic service routing agent.

Think of evaluation like a report card — it tells us
exactly where the agent is doing well and where it needs improvement.
We measure four things: accuracy, format, reasoning quality, and source presence.
"""

import json
import logging
import re
from typing import Any

import pandas as pd

from src.retrieval import keyword_retrieve
from src.schema import validate_response, VALID_JURISDICTION_LEVELS

logger = logging.getLogger(__name__)


# ── Keyword baseline (non-LLM) ─────────────────────────────────────────────────

def keyword_baseline_predict(question: str, catalog: pd.DataFrame) -> dict:
    """
    Simple keyword-matching baseline — no LLM required.

    Finds the best-matching catalog row by word overlap
    and returns its fields as a prediction.
    Used as the comparison baseline in evaluation.
    """
    from src.retrieval import tokenize
    q_tokens = set(tokenize(question))
    best_score = -1
    best_row = None

    for _, row in catalog.iterrows():
        text = " ".join([
            str(row.get("service_name", "")),
            str(row.get("description", "")),
            str(row.get("keywords", "")),
            str(row.get("responsible_body", "")),
        ]).lower()
        score = sum(tok in text for tok in q_tokens)
        if score > best_score:
            best_score = score
            best_row = row

    if best_row is None:
        return {
            "service_name": "Unknown",
            "jurisdiction_level": "Unclear",
            "responsible_body": "Unknown",
            "confidence": 0.0,
            "reasoning_summary": "No match found in catalog.",
            "next_steps": ["Contact 311 for guidance."],
            "sources": [],
        }

    return {
        "service_name": str(best_row.get("service_name", "")),
        "jurisdiction_level": str(best_row.get("jurisdiction_level", "Unclear")),
        "responsible_body": str(best_row.get("responsible_body", "")),
        "confidence": min(1.0, best_score / max(len(q_tokens), 1)),
        "reasoning_summary": str(best_row.get("description", "")),
        "next_steps": [str(best_row.get("next_steps_hint", ""))],
        "sources": [str(best_row.get("source_url", ""))],
    }


# ── Individual metrics ─────────────────────────────────────────────────────────

def format_compliance_check(response: dict) -> bool:
    """
    Check whether a response satisfies the output schema.

    Returns True if all required fields are present and valid.
    """
    is_valid, _ = validate_response(response)
    return is_valid


def reasoning_quality_score(response: dict) -> float:
    """
    Score the quality of the reasoning_summary on a 0.0–1.0 scale.

    Scoring rubric:
    - 0.0: Missing or empty reasoning
    - 0.3: Very short (< 20 chars)
    - 0.6: Mentions service name or jurisdiction
    - 0.8: Mentions responsible body
    - 1.0: Mentions service, jurisdiction, body, and >= 2 sentences

    This is a heuristic proxy — real scoring would use human judges
    or a second LLM as evaluator.
    """
    summary = response.get("reasoning_summary", "")
    if not summary or not summary.strip():
        return 0.0

    score = 0.0

    # Length check
    if len(summary) < 20:
        return 0.3

    score += 0.3  # base credit for non-trivial text

    # Mentions jurisdiction level
    jurisdiction = response.get("jurisdiction_level", "")
    if jurisdiction and jurisdiction.lower() in summary.lower():
        score += 0.2

    # Mentions responsible body
    body = response.get("responsible_body", "")
    if body and len(body) > 3 and body.lower() in summary.lower():
        score += 0.2

    # At least 2 sentences
    sentences = [s.strip() for s in re.split(r"[.!?]", summary) if s.strip()]
    if len(sentences) >= 2:
        score += 0.2

    # Mentions service name
    service = response.get("service_name", "")
    if service and service.lower() in summary.lower():
        score += 0.1

    return min(1.0, score)


def source_presence_rate(sources: list) -> bool:
    """Return True if at least one non-empty source URL is present."""
    return bool(sources) and any(str(s).strip() for s in sources)


# ── Single-row evaluation ──────────────────────────────────────────────────────

def evaluate_single(
    prediction: dict,
    expected_jurisdiction: str,
    expected_body: str,
) -> dict:
    """
    Evaluate a single prediction against ground truth.

    Returns a dict with all four metric scores:
    - jurisdiction_correct: exact match on jurisdiction level
    - body_correct: exact string match on responsible body
    - format_compliant: all schema fields present and valid
    - reasoning_quality: heuristic score 0.0–1.0
    - source_present: at least one source URL
    """
    return {
        "jurisdiction_correct": (
            prediction.get("jurisdiction_level", "").strip()
            == expected_jurisdiction.strip()
        ),
        "body_correct": (
            prediction.get("responsible_body", "").strip().lower()
            == expected_body.strip().lower()
        ),
        "format_compliant": format_compliance_check(prediction),
        "reasoning_quality": reasoning_quality_score(prediction),
        "source_present": source_presence_rate(prediction.get("sources", [])),
    }


# ── Batch evaluation ───────────────────────────────────────────────────────────

def evaluate_all(
    predictions: list[dict],
    eval_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Evaluate a list of predictions against the full evaluation set.

    Args:
        predictions: List of prediction dicts (one per eval row).
        eval_df:     DataFrame with columns: question, expected_jurisdiction_level,
                     expected_responsible_body, expected_service_name.

    Returns:
        DataFrame combining predictions with evaluation scores.
    """
    records = []
    for pred, (_, row) in zip(predictions, eval_df.iterrows()):
        scores = evaluate_single(
            pred,
            expected_jurisdiction=str(row["expected_jurisdiction_level"]),
            expected_body=str(row["expected_responsible_body"]),
        )
        record = {
            "question": row["question"],
            "expected_service": row.get("expected_service_name", ""),
            "expected_jurisdiction": row["expected_jurisdiction_level"],
            "expected_body": row["expected_responsible_body"],
            "predicted_service": pred.get("service_name", ""),
            "predicted_jurisdiction": pred.get("jurisdiction_level", ""),
            "predicted_body": pred.get("responsible_body", ""),
            "confidence": pred.get("confidence", 0.0),
            **scores,
        }
        records.append(record)

    return pd.DataFrame(records)


# ── Aggregate metrics ──────────────────────────────────────────────────────────

def compute_metrics(results_df: pd.DataFrame) -> dict:
    """
    Compute aggregate evaluation metrics across all predictions.

    Returns a dict ready for JSON serialization and DVC metrics tracking.
    """
    return {
        "jurisdiction_accuracy": float(results_df["jurisdiction_correct"].mean()),
        "responsible_body_accuracy": float(results_df["body_correct"].mean()),
        "format_compliance_rate": float(results_df["format_compliant"].mean()),
        "avg_reasoning_quality": float(results_df["reasoning_quality"].mean()),
        "source_presence_rate": float(results_df["source_present"].mean()),
        "n_examples": int(len(results_df)),
    }


# ── Scoring rubric display ─────────────────────────────────────────────────────

SCORING_RUBRIC = pd.DataFrame(
    [
        {
            "Metric": "jurisdiction_accuracy",
            "What It Measures": "Did we predict the right level of government?",
            "Full Credit (1.0)": "Exact match (e.g. 'Region' == 'Region')",
            "No Credit (0.0)": "Wrong level (e.g. 'City' when answer is 'Province')",
        },
        {
            "Metric": "responsible_body_accuracy",
            "What It Measures": "Did we name the right government body?",
            "Full Credit (1.0)": "Exact name match (case-insensitive)",
            "No Credit (0.0)": "Different body or missing",
        },
        {
            "Metric": "format_compliance_rate",
            "What It Measures": "Is the output JSON valid and complete?",
            "Full Credit (1.0)": "All 7 required fields present and valid types",
            "No Credit (0.0)": "Missing field or wrong jurisdiction_level value",
        },
        {
            "Metric": "avg_reasoning_quality",
            "What It Measures": "Is the reasoning explanation useful?",
            "Full Credit (1.0)": "Mentions service, body, jurisdiction; ≥ 2 sentences",
            "No Credit (0.0)": "Empty or < 20 chars",
        },
        {
            "Metric": "source_presence_rate",
            "What It Measures": "Did the agent cite at least one source URL?",
            "Full Credit (1.0)": "At least one non-empty URL in sources list",
            "No Credit (0.0)": "Empty sources list",
        },
    ]
)
