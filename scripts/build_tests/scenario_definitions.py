# scripts/scenario_definitions.py

from __future__ import annotations

import json
import re
from typing import List, Dict, Any

from openai import OpenAI
from dotenv import load_dotenv
load_dotenv()

from scripts.build_tests.edge_case_mutations import SCENARIO_REGISTRY


# =====================================================================
# LLM CLIENT
# =====================================================================
client = OpenAI()


# =====================================================================
# JSON EXTRACTION
# =====================================================================

def extract_json_from_text(text: str) -> dict:
    """Extract JSON from messy LLM output."""
    try:
        return json.loads(text)
    except:
        pass

    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not match:
        raise ValueError("No JSON object found")

    block = match.group(0)

    # Cleanup
    block = re.sub(r"[\x00-\x1F\x7F]", "", block)
    block = re.sub(r",\s*([}\]])", r"\1", block)
    block = block.replace("“", "\"").replace("”", "\"")

    return json.loads(block)


# =====================================================================
# MULTI-TABLE MUTATION HELPERS
# =====================================================================

def generate_join_keys(n: int, prefix: str = "jk") -> list[str]:
    """Generate matching join keys with row-level variation."""
    return [f"{prefix}_{i:05d}" for i in range(n)]


def mutate_join_only(dataset):
    """
    Deterministic scenario:
    - Ensures at least one table-to-table join produces real matches.
    - Works even though generator passes one DF at a time (then we no-op).
    """

    # --- CASE 1: Called on a single table (DataFrame) -> NO-OP ---
    # The synthetic generator applies inject(df) for each table individually.
    import pandas as pd

    if isinstance(dataset, pd.DataFrame):
        # Skip — join logic needs the full dataset, handled at top-level
        return dataset

    # --- CASE 2: Called on full dataset (dict of tables) ---
    # This happens only when generate_tests.py applies scenario
    if not isinstance(dataset, dict):
        return dataset

    # Deep copy
    new = {k: v.copy() for k, v in dataset.items()}

    # Need at least two tables
    if len(new) < 2:
        return new

    # Try common join columns
    join_candidates = ["CDSCode", "cdscode", "id"]

    join_col = None
    for col in join_candidates:
        if all(col in df.columns for df in new.values()):
            join_col = col
            break

    if join_col is None:
        return new

    # Generate matching keys
    n = min(len(df) for df in new.values())
    keys = [f"join_{i:05d}" for i in range(n)]

    # Assign matching join keys
    for table, df in new.items():
        df.loc[:n-1, join_col] = keys

    return new


# =====================================================================
# DETERMINISTIC SCENARIOS
# =====================================================================

def _deterministic_scenarios(limit: int | None) -> List[Dict[str, Any]]:
    """
    These are scenario *structures*.
    They do not mutate data until applied inside generate_tests.py.
    """
    scenarios = []

    # Only scenarios dependent on LIMIT
    if limit:
        scenarios.extend([
            {"name": "happy_path", "positive_count": limit + 2, "ordering": "correct"},
            {"name": "exact_limit", "positive_count": limit, "ordering": "correct"},
            {"name": "under_limit", "positive_count": max(1, limit - 1), "ordering": "correct"},
            {"name": "reverse_order", "positive_count": limit + 2, "ordering": "reverse"},
        ])

    # No-match scenario
    scenarios.append({"name": "no_match", "positive_count": 0})

    # JOIN scenario — NOW has real mutation function
    scenarios.append({
        "name": "join_only",
        "positive_count": 0,
        "inject": mutate_join_only
    })

    return scenarios


# =====================================================================
# LLM PROMPT FOR EDGE-CASE SCENARIOS
# =====================================================================

EDGE_CASE_LIST = "\n".join(f"- {name}" for name in sorted(SCENARIO_REGISTRY.keys()))

LLM_PROMPT_TEMPLATE = f"""
Your task is to choose 2–4 **edge-case data mutation scenarios**
from the following library:

{EDGE_CASE_LIST}

Output **only JSON**:

{{
  "selected_edge_cases": [
      "edge_case_name_1",
      "edge_case_name_2"
  ]
}}

RULES:
- Use only names shown above.
- Select 2–4 high-value cases.
- Do NOT use deterministic scenarios:
  ["happy_path", "exact_limit", "under_limit", "reverse_order", "no_match", "join_only"].
- No explanation, only JSON.

SQL QUERY TO ANALYZE:
"""


def _query_llm_for_edge_cases(sql: str) -> List[str]:
    prompt = LLM_PROMPT_TEMPLATE + sql

    try:
        resp = client.chat.completions.create(
            model="gpt-4o",
            temperature=0,
            max_tokens=300,
            messages=[{"role": "user", "content": prompt}]
        )

        raw = resp.choices[0].message.content.strip()
        obj = extract_json_from_text(raw)
        names = obj.get("selected_edge_cases", [])

        return [n for n in names if n in SCENARIO_REGISTRY]

    except Exception as e:
        print(f"[LLM Edge-Case Selector] Error: {e}")
        return []


# =====================================================================
# COMBINE DETERMINISTIC + LLM EDGE-CASES
# =====================================================================

def build_scenarios(sql: str, limit: int | None):
    """
    Returns a list like:
    [
        {name:"happy_path", positive_count:X, ...},
        {name:"join_only", inject:<fn>},
        {name:"large_numbers", inject:<fn>},
        ...
    ]
    """
    # Deterministic baseline scenarios
    scenarios = _deterministic_scenarios(limit)

    # LLM-selected high-value mutations
    chosen = _query_llm_for_edge_cases(sql)

    print(f"[Scenario Planner] Deterministic={len(scenarios)}, LLM edge-cases={len(chosen)}")

    # Convert to executable scenario objects
    for name in chosen:
        entry = SCENARIO_REGISTRY.get(name)
        if not entry:
            continue

        mutate_fn = entry.get("mutate")
        if not mutate_fn:
            continue

        scenarios.append({
            "name": name,
            "inject": mutate_fn
        })

    return scenarios
