"""
Stage 2 of the DVC pipeline: run evaluation on the service routing agent.

Loads the cleaned catalog and evaluation set, runs the keyword baseline
predictor (no API key needed — works fully offline), scores all predictions,
and writes artifacts/eval_predictions.json and artifacts/metrics.json.
"""

import json
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd
from src.retrieval import load_catalog
from src.evaluation import keyword_baseline_predict, evaluate_all, compute_metrics

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    catalog_path = ROOT / "artifacts" / "service_catalog.cleaned.json"
    eval_path = ROOT / "eval" / "service_eval_set.csv"
    preds_path = ROOT / "artifacts" / "eval_predictions.json"
    metrics_path = ROOT / "artifacts" / "metrics.json"

    logger.info("Loading catalog from %s", catalog_path)
    catalog = load_catalog(catalog_path)
    logger.info("Catalog: %d services", len(catalog))

    logger.info("Loading evaluation set from %s", eval_path)
    eval_df = pd.read_csv(eval_path)
    logger.info("Evaluation set: %d questions", len(eval_df))

    # Generate predictions using keyword baseline (reproducible, no API needed)
    logger.info("Generating keyword-baseline predictions...")
    predictions = [
        keyword_baseline_predict(str(row["question"]), catalog)
        for _, row in eval_df.iterrows()
    ]

    # Score predictions against ground truth
    results_df = evaluate_all(predictions, eval_df)

    # Compute aggregate metrics
    metrics = compute_metrics(results_df)
    logger.info("Metrics: %s", json.dumps(metrics, indent=2))

    # Save outputs
    results_df.to_json(preds_path, orient="records", indent=2)
    logger.info("Predictions written to %s", preds_path)

    metrics_path.write_text(json.dumps(metrics, indent=2))
    logger.info("Metrics written to %s", metrics_path)


if __name__ == "__main__":
    main()
