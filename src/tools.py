"""
Tool functions for the Claude tool-calling agent (Tier 3).

Each function here is a 'tool' the AI can ask to run —
like a calculator app on your phone that the AI can press buttons on.
The AI describes what it wants to look up; we run the code; we give it back the answer.
"""

import pandas as pd


# ── Tool implementations ───────────────────────────────────────────────────────

def search_service_index(query: str, catalog: pd.DataFrame, top_k: int = 3) -> list[dict]:
    """
    Search the local service catalog for entries relevant to the query.

    Args:
        query:   Resident's question or search phrase.
        catalog: The normalized service catalog DataFrame.
        top_k:   Number of results to return.

    Returns:
        List of matching catalog records as dicts.
    """
    from src.retrieval import keyword_retrieve
    results = keyword_retrieve(query, catalog, top_k=top_k)
    cols = ["service_name", "jurisdiction_level", "responsible_body",
            "description", "next_steps_hint", "source_url"]
    available_cols = [c for c in cols if c in results.columns]
    return results[available_cols].to_dict(orient="records")


def lookup_service_owner(service_name: str, catalog: pd.DataFrame) -> dict:
    """
    Look up the responsible level of government and body for an exact service name.

    Args:
        service_name: Canonical service name (e.g. "garbage pickup").
        catalog:      The normalized service catalog DataFrame.

    Returns:
        Dict with jurisdiction, responsible body, reasoning, and sources.
    """
    matches = catalog[
        catalog["service_name"].str.lower() == service_name.lower()
    ]
    if len(matches) == 0:
        # No exact match — return an 'unclear' signal
        return {
            "service_name": service_name,
            "jurisdiction_level": "Unclear",
            "responsible_body": "Unknown",
            "reasoning_summary": "No exact service match found in the local catalog.",
            "sources": [],
        }
    row = matches.iloc[0]
    return {
        "service_name": row["service_name"],
        "jurisdiction_level": row["jurisdiction_level"],
        "responsible_body": row["responsible_body"],
        "reasoning_summary": str(row.get("description", "")),
        "sources": [str(row.get("source_url", ""))] if row.get("source_url") else [],
    }


def suggest_next_steps(service_name: str, catalog: pd.DataFrame) -> dict:
    """
    Retrieve the recommended next steps for a service after ownership is known.

    Args:
        service_name: Canonical service name.
        catalog:      The normalized service catalog DataFrame.

    Returns:
        Dict with a list of next_steps strings.
    """
    matches = catalog[
        catalog["service_name"].str.lower() == service_name.lower()
    ]
    if len(matches) == 0:
        return {
            "next_steps": [
                "Search the relevant official government website for this service.",
                "Contact 311 (city) or the Region of Waterloo main line for guidance.",
            ]
        }
    row = matches.iloc[0]
    steps = [str(row.get("next_steps_hint", "Contact the relevant government office."))]
    if row.get("source_url"):
        steps.append(f"Verify details at the official source: {row['source_url']}")
    return {"next_steps": steps}


# ── Anthropic tool declarations ────────────────────────────────────────────────
# These are passed to client.messages.create(tools=TOOL_DECLARATIONS)

TOOL_DECLARATIONS = [
    {
        "name": "search_service_index",
        "description": (
            "Search the local municipal service catalog for entries relevant to a resident's query. "
            "Use this first to discover which services might match the question."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The resident's question or key service keywords.",
                }
            },
            "required": ["query"],
        },
    },
    {
        "name": "lookup_service_owner",
        "description": (
            "Look up the responsible level of government and body for a specific, known service name. "
            "Use this after search_service_index identifies the likely service name."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "service_name": {
                    "type": "string",
                    "description": "The canonical service name as it appears in the catalog (e.g. 'garbage pickup').",
                }
            },
            "required": ["service_name"],
        },
    },
    {
        "name": "suggest_next_steps",
        "description": (
            "Retrieve the recommended next steps a resident should take for a given service. "
            "Use this after confirming the service owner to provide actionable guidance."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "service_name": {
                    "type": "string",
                    "description": "The canonical service name.",
                }
            },
            "required": ["service_name"],
        },
    },
]


# ── Tool registry ──────────────────────────────────────────────────────────────

def make_tool_registry(catalog: pd.DataFrame) -> dict:
    """
    Build a mapping from tool name -> callable, with catalog pre-bound.

    This is the 'dispatcher' — when Claude says 'run search_service_index',
    we look up the function here and call it with the right arguments.
    """
    return {
        "search_service_index": lambda **kwargs: search_service_index(catalog=catalog, **kwargs),
        "lookup_service_owner": lambda **kwargs: lookup_service_owner(catalog=catalog, **kwargs),
        "suggest_next_steps": lambda **kwargs: suggest_next_steps(catalog=catalog, **kwargs),
    }
