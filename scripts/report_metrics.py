"""
Stage 3 of the DVC pipeline: generate a human-readable evaluation report.

Reads artifacts/metrics.json and writes artifacts/metrics_report.md.
"""

import json
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def format_pct(value: float) -> str:
    return f"{value * 100:.1f}%"


def main() -> None:
    metrics_path = ROOT / "artifacts" / "metrics.json"
    report_path = ROOT / "artifacts" / "metrics_report.md"

    if not metrics_path.exists():
        logger.error("metrics.json not found at %s — run run_agent_eval.py first", metrics_path)
        sys.exit(1)

    metrics = json.loads(metrics_path.read_text())
    logger.info("Loaded metrics: %s", metrics)

    # Build markdown report
    report_lines = [
        "# AI Agent Workshop — Evaluation Report",
        "",
        "## Summary",
        "",
        "| Metric | Score |",
        "|--------|-------|",
        f"| Jurisdiction Accuracy | {format_pct(metrics.get('jurisdiction_accuracy', 0))} |",
        f"| Responsible Body Accuracy | {format_pct(metrics.get('responsible_body_accuracy', 0))} |",
        f"| Format Compliance Rate | {format_pct(metrics.get('format_compliance_rate', 0))} |",
        f"| Avg Reasoning Quality | {metrics.get('avg_reasoning_quality', 0):.3f} / 1.000 |",
        f"| Source Presence Rate | {format_pct(metrics.get('source_presence_rate', 0))} |",
        f"| Number of Examples | {metrics.get('n_examples', 0)} |",
        "",
        "## Interpretation",
        "",
        "- **Jurisdiction Accuracy**: % of questions where the predicted level of government matches ground truth.",
        "- **Responsible Body Accuracy**: % where the exact government body name matches.",
        "- **Format Compliance Rate**: % of responses that pass full JSON schema validation.",
        "- **Avg Reasoning Quality**: Heuristic score (0-1) based on explanation completeness.",
        "- **Source Presence Rate**: % of responses that include at least one source URL.",
        "",
        "_Generated automatically by scripts/report_metrics.py_",
    ]

    report = "\n".join(report_lines)
    report_path.write_text(report, encoding="utf-8")
    logger.info("Report written to %s", report_path)
    print(report)


if __name__ == "__main__":
    main()
