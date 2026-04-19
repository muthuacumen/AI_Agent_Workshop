"""
Stage 1 of the DVC pipeline: normalize the service catalog.

Reads data/service_catalog.csv, cleans it, and writes
artifacts/service_catalog.cleaned.json for downstream stages.
"""

import logging
import sys
from pathlib import Path

# Allow imports from project root
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd
from src.retrieval import normalize_catalog

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    data_dir = ROOT / "data"
    artifacts_dir = ROOT / "artifacts"
    artifacts_dir.mkdir(exist_ok=True)

    input_path = data_dir / "service_catalog.csv"
    output_path = artifacts_dir / "service_catalog.cleaned.json"

    logger.info("Reading catalog from %s", input_path)
    df = pd.read_csv(input_path)
    logger.info("Loaded %d rows", len(df))

    # Validate required columns
    required = ["service_name", "jurisdiction_level", "responsible_body",
                "description", "keywords", "next_steps_hint", "source_url"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        logger.error("Missing columns: %s", missing)
        sys.exit(1)

    # Validate jurisdiction values
    valid_jurisdictions = {"City", "Region", "Province", "Federal", "Mixed", "Unclear"}
    bad_rows = df[~df["jurisdiction_level"].isin(valid_jurisdictions)]
    if len(bad_rows) > 0:
        logger.warning(
            "%d rows have unexpected jurisdiction_level values: %s",
            len(bad_rows),
            bad_rows["jurisdiction_level"].unique().tolist(),
        )

    # Normalize
    clean_df = normalize_catalog(df)
    logger.info("Normalization complete — added: service_name_normalized, keywords_list, retrieval_text")

    # Export
    clean_df.to_json(output_path, orient="records", indent=2)
    logger.info("Wrote %d records to %s", len(clean_df), output_path)


if __name__ == "__main__":
    main()
