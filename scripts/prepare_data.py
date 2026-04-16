import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
data_dir = ROOT / "data"
artifacts_dir = ROOT / "artifacts"
artifacts_dir.mkdir(exist_ok=True)

df = pd.read_csv(data_dir / "service_catalog.csv")
df["service_name_normalized"] = df["service_name"].str.lower().str.strip()
df["keywords_list"] = df["keywords"].fillna("").apply(lambda x: [k.strip() for k in x.split(";") if k.strip()])
df["retrieval_text"] = (
    df["service_name"].fillna("") + " | " +
    df["description"].fillna("") + " | " +
    df["keywords"].fillna("")
).str.lower()
df.to_json(artifacts_dir / "service_catalog.cleaned.json", orient="records", indent=2)
print(f"Wrote {artifacts_dir / 'service_catalog.cleaned.json'}")