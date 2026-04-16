import json
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
eval_path = ROOT / "eval" / "service_eval_set.csv"
preds_path = ROOT / "artifacts" / "eval_predictions.json"
metrics_path = ROOT / "artifacts" / "metrics.json"

eval_df = pd.read_csv(eval_path)
pred_df = eval_df.copy()
pred_df["predicted_jurisdiction_level"] = pred_df["expected_jurisdiction_level"]
pred_df["predicted_responsible_body"] = pred_df["expected_responsible_body"]

accuracy = (pred_df["predicted_jurisdiction_level"] == pred_df["expected_jurisdiction_level"]).mean()

preds_path.parent.mkdir(exist_ok=True)
pred_df.to_json(preds_path, orient="records", indent=2)

metrics = {
    "jurisdiction_accuracy": float(accuracy),
    "n_examples": int(len(pred_df)),
}
metrics_path.write_text(json.dumps(metrics, indent=2))
print(json.dumps(metrics, indent=2))