"""
End-to-end production pipeline for the civic service routing agent.

This module wires together all components into a single callable function:
  Input question
    → Preprocessing (normalize, retrieve)
    → Agent (Tier 1 / 2 / 3)
    → Post-processing (parse, validate)
    → Evaluation (score against ground truth if available)
    → Final output (structured response + audit log)

Think of it like a factory assembly line — each station does one job,
and the product moves through in order.
"""

import json
import logging
import time
from pathlib import Path
from typing import Any

import pandas as pd

from src.retrieval import load_catalog, keyword_retrieve
from src.schema import validate_response, ServiceResponse
from src.evaluation import evaluate_single, compute_metrics

logger = logging.getLogger(__name__)


# ── Pipeline config ────────────────────────────────────────────────────────────

DEFAULT_CATALOG_PATH = Path(__file__).resolve().parents[1] / "artifacts" / "service_catalog.cleaned.json"
DEFAULT_MODEL = "claude-sonnet-4-6"


# ── Pipeline runner ────────────────────────────────────────────────────────────

def run_pipeline(
    question: str,
    catalog: pd.DataFrame | None = None,
    catalog_path: str | Path = DEFAULT_CATALOG_PATH,
    tier: int = 3,
    model: str = DEFAULT_MODEL,
    expected_jurisdiction: str | None = None,
    expected_body: str | None = None,
) -> dict[str, Any]:
    """
    Run the full civic service routing pipeline for one question.

    Args:
        question:             Resident's question (free text).
        catalog:              Pre-loaded catalog DataFrame (optional — loaded from path if None).
        catalog_path:         Path to cleaned catalog JSON (used if catalog is None).
        tier:                 Agent tier: 1=baseline, 2=retrieval-augmented, 3=tool-calling.
        model:                Claude model name.
        expected_jurisdiction: Optional ground truth for on-the-fly evaluation.
        expected_body:         Optional ground truth for on-the-fly evaluation.

    Returns:
        Dict with keys:
          - response:      The ServiceResponse dict
          - is_valid:      Whether the response passes schema validation
          - errors:        List of schema validation errors (empty if valid)
          - eval_scores:   Evaluation scores dict (only if expected values provided)
          - latency_ms:    Wall-clock time for the agent call
          - tier_used:     Which tier was used
          - tool_trace:    List of tool calls made (Tier 3 only)
    """
    # ── Step 1: Load catalog ───────────────────────────────────────────────────
    if catalog is None:
        logger.info("Loading catalog from %s", catalog_path)
        catalog = load_catalog(catalog_path)

    # ── Step 2: Import agent (lazy to allow offline use) ──────────────────────
    try:
        from src.agent import make_client, run_agent, tool_agent_call
        client = make_client()
        api_available = True
    except (ImportError, EnvironmentError) as exc:
        logger.warning("Claude API unavailable (%s) — falling back to keyword baseline", exc)
        api_available = False

    # ── Step 3: Run agent (or fallback) ───────────────────────────────────────
    tool_trace: list[dict] = []
    start_time = time.perf_counter()

    if tier == 0 or not api_available:
        # Tier 0 = explicit keyword baseline (no LLM, works fully offline)
        from src.evaluation import keyword_baseline_predict
        response = keyword_baseline_predict(question, catalog)
        tier = 0
    elif tier == 3:
        response, tool_trace = tool_agent_call(question, catalog, client, model=model)
    else:
        response = run_agent(question, catalog, client, tier=tier, model=model)

    latency_ms = round((time.perf_counter() - start_time) * 1000, 1)

    # ── Step 4: Post-processing — validate response ────────────────────────────
    is_valid, errors = validate_response(response)
    if not is_valid:
        logger.warning("Response failed schema validation: %s", errors)

    # ── Step 5: Evaluate (if ground truth provided) ────────────────────────────
    eval_scores: dict | None = None
    if expected_jurisdiction is not None and expected_body is not None:
        eval_scores = evaluate_single(response, expected_jurisdiction, expected_body)

    # ── Step 6: Assemble output ────────────────────────────────────────────────
    return {
        "response": response,
        "is_valid": is_valid,
        "errors": errors,
        "eval_scores": eval_scores,
        "latency_ms": latency_ms,
        "tier_used": tier,
        "tool_trace": tool_trace,
    }


# ── Batch pipeline ─────────────────────────────────────────────────────────────

def run_pipeline_batch(
    eval_df: pd.DataFrame,
    catalog: pd.DataFrame,
    tier: int = 2,
    model: str = DEFAULT_MODEL,
) -> tuple[pd.DataFrame, dict]:
    """
    Run the pipeline on every row in an evaluation DataFrame.

    Args:
        eval_df:  DataFrame with columns: question, expected_jurisdiction_level,
                  expected_responsible_body.
        catalog:  Pre-loaded normalized catalog.
        tier:     Agent tier to use.
        model:    Claude model name.

    Returns:
        (results_df, metrics_dict)
    """
    try:
        from src.agent import make_client, run_agent, tool_agent_call
        client = make_client()
        api_available = True
    except (ImportError, EnvironmentError):
        api_available = False

    predictions = []
    for _, row in eval_df.iterrows():
        question = str(row["question"])
        logger.info("Running pipeline on: %s", question[:60])

        if tier == 0 or not api_available:
            from src.evaluation import keyword_baseline_predict
            pred = keyword_baseline_predict(question, catalog)
        elif tier == 3:
            pred, _ = tool_agent_call(question, catalog, client, model=model)
        else:
            pred = run_agent(question, catalog, client, tier=tier, model=model)

        predictions.append(pred)

    from src.evaluation import evaluate_all, compute_metrics
    results_df = evaluate_all(predictions, eval_df)
    metrics = compute_metrics(results_df)

    return results_df, metrics


# ── Pretty printer ─────────────────────────────────────────────────────────────

def print_response(result: dict[str, Any]) -> None:
    """Print a pipeline result in a readable format for notebooks."""
    response = result["response"]
    print("=" * 60)
    print(f"Service:      {response.get('service_name', 'N/A')}")
    print(f"Jurisdiction: {response.get('jurisdiction_level', 'N/A')}")
    print(f"Responsible:  {response.get('responsible_body', 'N/A')}")
    print(f"Confidence:   {response.get('confidence', 0):.0%}")
    print()
    print(f"Reasoning:    {response.get('reasoning_summary', '')}")
    print()
    steps = response.get("next_steps", [])
    print("Next Steps:")
    for i, step in enumerate(steps, 1):
        print(f"  {i}. {step}")
    sources = response.get("sources", [])
    if sources:
        print()
        print("Sources:")
        for src in sources:
            print(f"  • {src}")
    print()
    print(f"Valid schema: {'✓' if result['is_valid'] else '✗'}")
    print(f"Latency:      {result['latency_ms']} ms  |  Tier: {result['tier_used']}")
    if result.get("tool_trace"):
        print(f"Tool calls:   {len(result['tool_trace'])}")
        for call in result["tool_trace"]:
            print(f"  → {call['tool']}({call['args']})")
    print("=" * 60)
