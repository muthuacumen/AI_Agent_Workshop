"""
Claude-powered civic service routing agent.

This module provides three tiers of increasing sophistication:
  Tier 1 - Baseline:           Raw Claude call, no grounding
  Tier 2 - Retrieval-Augmented: Claude call with catalog context injected
  Tier 3 - Tool-Calling Agent:  Claude calls tools, we execute, Claude synthesises

Think of the tiers like asking for directions:
  Tier 1: Ask a stranger from memory (might be wrong)
  Tier 2: Ask a stranger while showing them a map (better)
  Tier 3: Ask an expert who can look things up in real time (best)
"""

import json
import logging
import os
import re
from typing import Any

import anthropic
import pandas as pd

from src.retrieval import build_system_prompt, build_grounded_prompt, keyword_retrieve
from src.schema import validate_response, ServiceResponse
from src.tools import TOOL_DECLARATIONS, make_tool_registry

logger = logging.getLogger(__name__)


# ── Client setup ───────────────────────────────────────────────────────────────

def make_client() -> anthropic.Anthropic:
    """Create an Anthropic client from the ANTHROPIC_API_KEY env var."""
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "ANTHROPIC_API_KEY is not set. "
            "Add it to your .env file or export it in your shell."
        )
    return anthropic.Anthropic(api_key=api_key)


# ── Response parsing ───────────────────────────────────────────────────────────

def parse_json_response(raw_text: str) -> dict:
    """
    Extract a JSON object from a Claude text response.

    Claude is instructed to return only JSON, but sometimes
    wraps it in markdown fences. This strips those out.
    """
    # Strip markdown code fences if present
    cleaned = re.sub(r"```(?:json)?\s*", "", raw_text).strip().rstrip("`").strip()

    # Find the outermost JSON object
    start = cleaned.find("{")
    end = cleaned.rfind("}") + 1
    if start == -1 or end == 0:
        raise ValueError(f"No JSON object found in response:\n{raw_text[:200]}")

    json_str = cleaned[start:end]
    return json.loads(json_str)


# ── Tier 1: Baseline ───────────────────────────────────────────────────────────

def baseline_call(
    question: str,
    client: anthropic.Anthropic,
    model: str = "claude-sonnet-4-6",
) -> dict:
    """
    Tier 1 — Pure LLM call with no grounding data.

    The agent relies entirely on its training knowledge.
    Fast, but may hallucinate jurisdiction details for local services.
    """
    system_prompt = build_system_prompt(catalog_context=None)

    message = client.messages.create(
        model=model,
        max_tokens=1024,
        system=system_prompt,
        messages=[{"role": "user", "content": question}],
    )

    raw_text = message.content[0].text
    logger.debug("Tier 1 raw response: %s", raw_text[:300])

    try:
        return parse_json_response(raw_text)
    except (ValueError, json.JSONDecodeError) as exc:
        logger.warning("Tier 1 JSON parse failed: %s", exc)
        return {
            "service_name": "Unknown",
            "jurisdiction_level": "Unclear",
            "responsible_body": "Unknown",
            "confidence": 0.0,
            "reasoning_summary": f"Parse error: {exc}",
            "next_steps": ["Please rephrase your question and try again."],
            "sources": [],
        }


# ── Tier 2: Retrieval-Augmented ────────────────────────────────────────────────

def grounded_call(
    question: str,
    catalog: pd.DataFrame,
    client: anthropic.Anthropic,
    model: str = "claude-sonnet-4-6",
    top_k: int = 3,
) -> dict:
    """
    Tier 2 — Retrieval-Augmented Generation (RAG).

    We first search the catalog for relevant entries,
    then inject them into Claude's context.
    Claude answers based on real catalog data — far less hallucination.

    Prompt caching is applied to the system prompt (which contains the
    catalog context) so repeated calls with the same context are cheaper.
    """
    retrieved_df = keyword_retrieve(question, catalog, top_k=top_k)
    catalog_context = retrieved_df[
        ["service_name", "jurisdiction_level", "responsible_body",
         "description", "next_steps_hint", "source_url"]
    ].to_dict(orient="records")

    # Build system prompt with catalog embedded — mark for caching
    system_prompt = build_system_prompt(catalog_context=catalog_context)

    message = client.messages.create(
        model=model,
        max_tokens=1024,
        system=[
            {
                "type": "text",
                "text": system_prompt,
                # Cache the system prompt — catalog text rarely changes
                "cache_control": {"type": "ephemeral"},
            }
        ],
        messages=[{"role": "user", "content": question}],
    )

    raw_text = message.content[0].text
    logger.debug("Tier 2 raw response: %s", raw_text[:300])

    try:
        return parse_json_response(raw_text)
    except (ValueError, json.JSONDecodeError) as exc:
        logger.warning("Tier 2 JSON parse failed: %s", exc)
        return {
            "service_name": "Unknown",
            "jurisdiction_level": "Unclear",
            "responsible_body": "Unknown",
            "confidence": 0.0,
            "reasoning_summary": f"Parse error: {exc}",
            "next_steps": ["Please rephrase your question and try again."],
            "sources": [],
        }


# ── Tier 3: Tool-Calling Agent ─────────────────────────────────────────────────

def tool_agent_call(
    question: str,
    catalog: pd.DataFrame,
    client: anthropic.Anthropic,
    model: str = "claude-sonnet-4-6",
    max_tool_rounds: int = 5,
) -> tuple[dict, list[dict]]:
    """
    Tier 3 — Full agentic loop with tool calling.

    Claude decides WHICH tools to call and WHEN.
    We execute the tools and feed results back.
    This continues until Claude produces a final JSON answer.

    Returns:
        (parsed_response_dict, tool_trace)
        tool_trace is a list of {"tool": name, "args": ..., "result": ...}
        for inspection / explanation of the agent's reasoning steps.
    """
    tool_registry = make_tool_registry(catalog)
    tool_trace: list[dict] = []

    # System prompt — no catalog context here (Claude will retrieve via tools)
    system_prompt = build_system_prompt(catalog_context=None)

    # Build the conversation messages list (grows as tools are called)
    messages: list[dict] = [{"role": "user", "content": question}]

    for round_num in range(max_tool_rounds):
        response = client.messages.create(
            model=model,
            max_tokens=2048,
            system=[
                {
                    "type": "text",
                    "text": system_prompt,
                    "cache_control": {"type": "ephemeral"},
                }
            ],
            tools=TOOL_DECLARATIONS,
            messages=messages,
        )

        logger.debug("Round %d stop_reason: %s", round_num + 1, response.stop_reason)

        if response.stop_reason == "end_turn":
            # Claude finished — extract the text answer
            for block in response.content:
                if hasattr(block, "text"):
                    try:
                        result = parse_json_response(block.text)
                        return result, tool_trace
                    except (ValueError, json.JSONDecodeError) as exc:
                        logger.warning("Final parse failed: %s", exc)
            break

        if response.stop_reason == "tool_use":
            # Claude wants to call one or more tools
            assistant_message: dict[str, Any] = {
                "role": "assistant",
                "content": response.content,
            }
            messages.append(assistant_message)

            # Execute all tool calls in this round
            tool_results = []
            for block in response.content:
                if block.type != "tool_use":
                    continue

                tool_name = block.name
                tool_args = dict(block.input)
                logger.info("Tool call: %s(%s)", tool_name, tool_args)

                if tool_name not in tool_registry:
                    tool_result = {"error": f"Unknown tool: {tool_name}"}
                else:
                    try:
                        tool_result = tool_registry[tool_name](**tool_args)
                    except Exception as exc:
                        tool_result = {"error": str(exc)}
                        logger.error("Tool %s failed: %s", tool_name, exc)

                tool_trace.append({
                    "round": round_num + 1,
                    "tool": tool_name,
                    "args": tool_args,
                    "result": tool_result,
                })

                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": json.dumps(tool_result),
                })

            # Feed all tool results back to Claude
            messages.append({"role": "user", "content": tool_results})

        else:
            # Unexpected stop reason
            logger.warning("Unexpected stop_reason: %s", response.stop_reason)
            break

    # If we exit the loop without a valid response, return a fallback
    fallback: dict = {
        "service_name": "Unknown",
        "jurisdiction_level": "Unclear",
        "responsible_body": "Unknown",
        "confidence": 0.0,
        "reasoning_summary": "Agent could not produce a valid response within the allowed tool rounds.",
        "next_steps": ["Please rephrase your question and try again."],
        "sources": [],
    }
    return fallback, tool_trace


# ── Convenience wrapper ────────────────────────────────────────────────────────

def run_agent(
    question: str,
    catalog: pd.DataFrame,
    client: anthropic.Anthropic,
    tier: int = 3,
    model: str = "claude-sonnet-4-6",
) -> dict:
    """
    Run the service routing agent at a given tier (1, 2, or 3).

    Args:
        question: The resident's question.
        catalog:  Normalized service catalog DataFrame.
        client:   Anthropic client.
        tier:     1 = baseline, 2 = retrieval-augmented, 3 = tool-calling.
        model:    Claude model to use.

    Returns:
        Parsed ServiceResponse dict.
    """
    if tier == 1:
        return baseline_call(question, client, model=model)
    elif tier == 2:
        return grounded_call(question, catalog, client, model=model)
    elif tier == 3:
        result, _ = tool_agent_call(question, catalog, client, model=model)
        return result
    else:
        raise ValueError(f"tier must be 1, 2, or 3 — got {tier}")
