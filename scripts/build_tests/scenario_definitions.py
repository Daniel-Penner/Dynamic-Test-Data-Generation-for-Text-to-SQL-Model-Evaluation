# scripts/scenario_definitions.py

from __future__ import annotations

import json
import re
from typing import List, Dict, Any

from openai import OpenAI
from dotenv import load_dotenv
load_dotenv()

# NEW — your edge case registry
from scripts.build_tests.edge_case_mutations import SCENARIO_REGISTRY

# --------------------------------------------------------------------
# LLM Client
# --------------------------------------------------------------------

client = OpenAI()  # uses OPENAI_API_KEY


# --------------------------------------------------------------------
# JSON Extraction (unchanged)
# --------------------------------------------------------------------

def extract_json_from_text(text: str) -> dict:
    try:
        return json.loads(text)
    except:
        pass

    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not match:
        raise ValueError("No JSON object found")

    block = match.group(0)

    block = re.sub(r"[\x00-\x1F\x7F]", "", block)
    block = re.sub(r",\s*([}\]])", r"\1", block)
    block = block.replace("“", "\"").replace("”", "\"")

    return json.loads(block)


# --------------------------------------------------------------------
# Deterministic Scenarios
# --------------------------------------------------------------------

def _deterministic_scenarios(limit: int | None) -> List[Dict[str, Any]]:
    scenarios = []

    if limit:
        scenarios.extend([
            {"name": "happy_path", "positive_count": limit + 2, "ordering": "correct"},
            {"name": "exact_limit", "positive_count": limit, "ordering": "correct"},
            {"name": "under_limit", "positive_count": max(1, limit - 1), "ordering": "correct"},
            {"name": "reverse_order", "positive_count": limit + 2, "ordering": "reverse"},
        ])

    scenarios.append({"name": "no_match", "positive_count": 0})
    scenarios.append({"name": "join_only", "positive_count": 0, "force_join_match": True})

    return scenarios


# --------------------------------------------------------------------
# LLM Prompt – NOW INCLUDES THE EDGE CASE LIBRARY
# --------------------------------------------------------------------

EDGE_CASE_LIST = "\n".join(
    f"- {name}" for name in sorted(SCENARIO_REGISTRY.keys())
)

LLM_PROMPT_TEMPLATE = f"""
Your task is to choose the **most relevant edge-case data mutation scenarios**
from the following master library:

{EDGE_CASE_LIST}

These represent powerful test-case generators (data mutation functions)
that distort synthetic datasets to expose SQL query weaknesses.

You must output JSON of the form:

{{
  "selected_edge_cases": [
      "edge_case_name_1",
      "edge_case_name_2",
      ...
  ]
}}

RULES:
- Only use names from the list above.
- Select 2–4 high-value edge cases.
- Do NOT use deterministic baseline scenarios:
  ["happy_path", "exact_limit", "under_limit", "reverse_order", "no_match", "join_only"].
- Only output JSON, no explanations.

SQL QUERY TO ANALYZE:
"""


def _query_llm_for_edge_cases(sql: str) -> List[str]:
    prompt = LLM_PROMPT_TEMPLATE + sql

    try:
        resp = client.chat.completions.create(
            model="gpt-4o",
            temperature=0,
            max_tokens=500,
            messages=[{"role": "user", "content": prompt}]
        )

        raw = resp.choices[0].message.content.strip()
        obj = extract_json_from_text(raw)
        names = obj.get("selected_edge_cases", [])

        # keep only names that exist in the registry
        return [n for n in names if n in SCENARIO_REGISTRY]

    except Exception as e:
        print(f"[LLM Edge-Case Selector] Error: {e}")
        return []


# --------------------------------------------------------------------
# Combine Deterministic + LLM Edge-Case Scenarios
# --------------------------------------------------------------------

def build_scenarios(sql: str, limit: int | None):
    """
    Build deterministic + LLM-selected edge-case scenarios.
    Ensures that each edge-case registry entry returns only its mutate() function.
    """

    # 1. Deterministic scenarios (limit-based, ordering, NULL injections, etc.)
    base = _deterministic_scenarios(limit)

    # 2. GPT chooses appropriate edge-case labels
    chosen_edge_cases = _query_llm_for_edge_cases(sql)

    print(f"[Scenario Planner] Deterministic={len(base)}, LLM edge-cases={len(chosen_edge_cases)}")

    # 3. Convert edge-case labels → fully functional scenario objects
    for name in chosen_edge_cases:
        registry_entry = SCENARIO_REGISTRY.get(name)

        if registry_entry is None:
            print(f"[Scenario Planner] ⚠ Unknown edge-case name '{name}', skipping.")
            continue

        mutate_fn = registry_entry.get("mutate")
        if mutate_fn is None:
            print(f"[Scenario Planner] ⚠ Edge-case '{name}' has no 'mutate' function, skipping.")
            continue

        base.append({
            "name": name,
            "inject": mutate_fn      # ✓ FIXED
        })

    return base
