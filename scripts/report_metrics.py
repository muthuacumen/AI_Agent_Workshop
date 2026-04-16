import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
metrics_path = ROOT / "artifacts" / "metrics.json"
report_path = ROOT / "artifacts" / "metrics_report.md"

metrics = json.loads(metrics_path.read_text())
report = f"""# Day 2 Evaluation Report

- Jurisdiction accuracy: {metrics['jurisdiction_accuracy']:.3f}
- Number of examples: {metrics['n_examples']}
"""
report_path.write_text(report)
print(report)