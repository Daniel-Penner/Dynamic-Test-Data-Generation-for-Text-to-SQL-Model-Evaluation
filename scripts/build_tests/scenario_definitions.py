# scripts/scenario_definitions.py

from __future__ import annotations

import json
import re
from typing import List, Dict, Any
from openai import OpenAI
from dotenv import load_dotenv
load_dotenv()


# --------------------------------------------------------------------
# LLM Client
# --------------------------------------------------------------------

client = OpenAI()   # uses OPENAI_API_KEY from env

def extract_json_from_text(text: str) -> dict:
    """
    Extract the first valid JSON object from a text string.
    Handles:
      - leading/trailing commentary
      - unescaped quotes inside values
      - minor syntax issues
    """
    # Try direct parse first
    try:
        return json.loads(text)
    except:
        pass

    # Extract the largest {...} block
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not match:
        raise ValueError("No JSON object found")

    block = match.group(0)

    # Fix common JSON issues ------------------------------------

    # Remove illegal control characters
    block = re.sub(r"[\x00-\x1F\x7F]", "", block)

    # Remove trailing commas before } or ]
    block = re.sub(r",\s*([}\]])", r"\1", block)

    # Replace smart quotes with normal ones
    block = block.replace("“", "\"").replace("”", "\"").replace("‘", "'").replace("’", "'")

    # Escape unescaped double-quotes inside string values
    def escape_quotes(m):
        content = m.group(1)
        content = content.replace('"', '\\"')
        return f"\"{content}\""

    block = re.sub(r'"([^"]*?)"', escape_quotes, block)

    # Final attempt
    return json.loads(block)

# --------------------------------------------------------------------
# Deterministic Baseline Scenarios (your original ones)
# --------------------------------------------------------------------

def _deterministic_scenarios(limit: int | None) -> List[Dict[str, Any]]:
    scenarios = []

    if limit:
        scenarios.append({
            "name": "happy_path",
            "positive_count": limit + 2,
            "ordering": "correct"
        })

        scenarios.append({
            "name": "exact_limit",
            "positive_count": limit,
            "ordering": "correct"
        })

        scenarios.append({
            "name": "under_limit",
            "positive_count": max(1, limit - 1),
            "ordering": "correct"
        })

        scenarios.append({
            "name": "reverse_order",
            "positive_count": limit + 2,
            "ordering": "reverse"
        })

    scenarios.append({
        "name": "no_match",
        "positive_count": 0,
        "ordering": None
    })

    scenarios.append({
        "name": "join_only",
        "positive_count": 0,
        "force_join_match": True,
        "ordering": None
    })

    return scenarios


# --------------------------------------------------------------------
# LLM-based scenario generator
# --------------------------------------------------------------------

LLM_PROMPT = """
Your task is to analyze the following SQL query and generate **additional unit test scenarios**
that would meaningfully test its semantics.

You MUST produce tricky, edge-case, or corner-case situations that deterministic
rule-based systems would have trouble generating.

Examples include:
- join anomalies
- NULL behavior cases
- division by zero edge cases
- empty tables
- filter boundary conditions (x, x±1, x±epsilon)
- multi-join mismatch / overmatch
- cases involving DISTINCT, aggregates, subqueries
- grouping edge cases
- ordering ambiguities

CRITICAL RULES:
- You MUST respond in JSON.
- Each scenario must have a unique `"name"` string.
- You MUST NOT recreate the deterministic default scenarios:
  ["happy_path", "exact_limit", "under_limit", "reverse_order", "no_match", "join_only"].

Output format:
{
  "scenarios": [
    { "name": "...", "description": "...", "hints": { ... } },
    ...
  ]
}
"""

def _query_llm_for_scenarios(sql: str) -> List[Dict[str, Any]]:
    """
    Query the LLM for additional scenarios.
    Uses a JSON balloon technique to make output safely parseable.
    Compatible with all versions of openai-python.
    """

    balloon_prompt = f"""
    You MUST output ONLY valid JSON.

    STRICT RULES:
    - No commentary
    - No Markdown
    - No backticks
    - No explanation
    - Output exactly one JSON object
    - Keys and strings must use double quotes only

    REQUIRED JSON SCHEMA:
    {{
      "scenarios": [
        {{
          "name": "string",
          "description": "string"
        }}
      ]
    }}

    DO NOT include any of these reserved scenario names:
      ["happy_path", "exact_limit", "under_limit", "reverse_order", "no_match", "join_only"]

    Generate 3–6 additional edge-case scenarios for this SQL:
    {sql}
    """

    try:
        resp = client.chat.completions.create(
            model="gpt-4o",         # Or gpt-4.1, gpt-4o-mini, etc.
            messages=[{
                "role": "user",
                "content": balloon_prompt
            }],
            temperature=0,
            max_tokens=800
        )

        raw = resp.choices[0].message.content.strip()

        # Try parse directly
        try:
            obj = json.loads(raw)
        except:
            # Extract inner {...} region safely
            import re
            match = re.search(r"\{.*\}", raw, flags=re.DOTALL)
            if not match:
                raise ValueError("No JSON object found in LLM output")
            obj = json.loads(match.group(0))

        # Validate
        scenarios = obj.get("scenarios", [])
        if not isinstance(scenarios, list):
            return []

        return scenarios

    except Exception as e:
        print(f"[LLM Scenario Planner] Error: {e}")
        return []


# --------------------------------------------------------------------
# Merge LLM scenarios with deterministic ones
# --------------------------------------------------------------------

def _merge_scenarios(
    base: List[Dict[str, Any]],
    llm_scenarios: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:

    names = {sc["name"] for sc in base}  # prevent collisions
    merged = base.copy()

    for sc in llm_scenarios:
        name = sc.get("name")
        if not name or name in names:
            continue  # skip duplicates or invalid
        merged.append(sc)
        names.add(name)

    return merged


# --------------------------------------------------------------------
# Public API
# --------------------------------------------------------------------

def build_scenarios(sql: str, limit: int | None) -> List[Dict[str, Any]]:
    """
    Builds the combined scenario set:
      - deterministic baseline scenarios
      - optional LLM-enhanced scenarios

    Uses only "limit" for deterministic scenarios.
    Uses full SQL text for LLM reasoning.
    """

    base_scenarios = _deterministic_scenarios(limit)
    llm_scenarios = _query_llm_for_scenarios(sql)

    final = _merge_scenarios(base_scenarios, llm_scenarios)

    print(f"[Scenario Planner] Deterministic={len(base_scenarios)}, LLM={len(llm_scenarios)}, Total={len(final)}")

    return final
