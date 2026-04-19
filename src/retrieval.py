"""
Retrieval functions for the civic service routing agent.

Think of this like a search engine for a small library.
Given a resident's question, we find the most relevant
service catalog entries to give the agent useful context.
"""

import re
import json
import pandas as pd
from pathlib import Path


# ── Tokenization ───────────────────────────────────────────────────────────────

def tokenize(text: str) -> list[str]:
    """
    Break a string into individual words (tokens).
    Strips punctuation and lowercases everything.

    Example: "Garbage Pickup!" -> ["garbage", "pickup"]
    """
    return re.findall(r"[a-z0-9]+", text.lower())


# ── Catalog normalization ──────────────────────────────────────────────────────

def normalize_catalog(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and enrich the raw service catalog CSV.

    Adds three helper columns:
    - service_name_normalized: lowercase for exact matching
    - keywords_list: semicolon-split list for programmatic use
    - retrieval_text: one big searchable string per row
    """
    out = df.copy()
    out["service_name_normalized"] = out["service_name"].str.lower().str.strip()
    out["keywords_list"] = out["keywords"].fillna("").apply(
        lambda x: [k.strip().lower() for k in x.split(";") if k.strip()]
    )
    out["retrieval_text"] = (
        out["service_name"].fillna("") + " | "
        + out["description"].fillna("") + " | "
        + out["keywords"].fillna("")
        + out["responsible_body"].fillna("")
    ).str.lower()
    return out


def load_catalog(catalog_path: str | Path) -> pd.DataFrame:
    """Load and normalize the service catalog from a JSON or CSV file."""
    catalog_path = Path(catalog_path)
    if catalog_path.suffix == ".json":
        df = pd.read_json(catalog_path)
    else:
        df = pd.read_csv(catalog_path)
    return normalize_catalog(df)


# ── Keyword retrieval ──────────────────────────────────────────────────────────

def keyword_retrieve(query: str, df: pd.DataFrame, top_k: int = 3) -> pd.DataFrame:
    """
    Find the top_k most relevant catalog rows for a query.

    How it works: count how many words in the question
    also appear in each catalog row. The row with the most
    matches wins — like a word-overlap score.

    Args:
        query:  The resident's question (free text).
        df:     The normalized service catalog DataFrame.
        top_k:  How many results to return.

    Returns:
        DataFrame of the top_k matching rows.
    """
    q_tokens = set(tokenize(query))
    scored_rows = []

    for _, row in df.iterrows():
        text = str(row.get("retrieval_text", ""))
        row_tokens = set(tokenize(text))
        overlap = len(q_tokens & row_tokens)
        scored_rows.append((overlap, row.to_dict()))

    # Sort by overlap score, highest first; filter zero-match rows
    ranked = sorted(scored_rows, key=lambda x: x[0], reverse=True)
    ranked = [row for score, row in ranked if score > 0][:top_k]

    if not ranked:
        # Fallback: return first top_k rows if nothing matched
        return df.head(top_k)

    return pd.DataFrame(ranked)


# ── Prompt builders ────────────────────────────────────────────────────────────

def build_system_prompt(catalog_context: list[dict] | None = None) -> str:
    """
    Build the system prompt for the Claude agent.

    If catalog_context is provided (retrieved rows), it is embedded
    directly in the system prompt — this is where prompt caching
    gives us the biggest savings (catalog text rarely changes).
    """
    schema_str = json.dumps(
        {
            "service_name": "<canonical name from catalog>",
            "jurisdiction_level": "City | Region | Province | Federal | Mixed | Unclear",
            "responsible_body": "<exact government body name>",
            "confidence": "<float 0.0–1.0>",
            "reasoning_summary": "<1-3 sentence grounded explanation>",
            "next_steps": ["<action 1>", "<action 2>"],
            "sources": ["<url 1>"],
        },
        indent=2,
    )

    base = f"""You are a public-service routing assistant for residents in Kitchener and Waterloo Region, Ontario, Canada.

Your job is to identify which level of government (City, Region, Province, Federal, Mixed, or Unclear) \
is responsible for a resident's service request, and guide them toward the correct next steps.

STRICT RULES:
1. Always respond with a single valid JSON object — no markdown fences, no extra text.
2. The JSON must match this exact schema:
{schema_str}
3. jurisdiction_level MUST be exactly one of: City, Region, Province, Federal, Mixed, Unclear
4. confidence is a decimal between 0.0 and 1.0 (e.g. 0.92)
5. next_steps must have at least one item
6. sources should include official government URLs when available
7. If you are not sure, set jurisdiction_level to "Unclear" and confidence below 0.5
8. Base your answer on the retrieved catalog entries provided. Do not invent services or bodies."""

    if catalog_context:
        catalog_str = json.dumps(catalog_context, indent=2)
        base += f"""

RETRIEVED SERVICE CATALOG ENTRIES (use these as your primary evidence):
{catalog_str}"""

    return base


def build_grounded_prompt(question: str, retrieved_df: pd.DataFrame) -> str:
    """
    Build a user message that injects retrieved context.
    Used for the retrieval-augmented (Tier 2) approach.
    """
    context = retrieved_df[
        ["service_name", "jurisdiction_level", "responsible_body",
         "description", "next_steps_hint", "source_url"]
    ].to_dict(orient="records")

    context_str = json.dumps(context, indent=2)
    return (
        f"Retrieved context from service catalog:\n{context_str}\n\n"
        f"Resident question: {question}"
    )
