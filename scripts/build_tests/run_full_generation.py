# scripts/build_tests/run_full_generation.py

from __future__ import annotations

import json
import sys
from pathlib import Path

from scripts.build_tests.generate_tests import generate_tests_for_query
from scripts.build_tests.build_schema_map import build_schema_map


# Ensure project root is on sys.path
# Always set project root to the repository root containing THIS script
PROJECT_ROOT = Path(__file__).resolve().parents[2]
while PROJECT_ROOT.name not in {
    "Dynamic-Test-Data-Generation-for-Text-to-SQL-Model-Evaluation"
}:
    PROJECT_ROOT = PROJECT_ROOT.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

BIRD_PATH = PROJECT_ROOT / "bird_input_data"
QUERIES_PATH = BIRD_PATH / "dev.json"
TABLES_PATH = BIRD_PATH / "dev_tables.json"

OUTPUT_DIR = PROJECT_ROOT / "tests"


def load_bird():
    with open(QUERIES_PATH, "r", encoding="utf-8") as f:
        queries = json.load(f)
    with open(TABLES_PATH, "r", encoding="utf-8") as f:
        tables = json.load(f)
    return queries, tables


def find_schema_entry(tables_meta, db_id: str):
    for entry in tables_meta:
        if entry["db_id"] == db_id:
            return entry
    return None


def run_all_queries(limit: int):
    queries, tables_meta = load_bird()

    for i, q in enumerate(queries[:limit]):
        db_id = q["db_id"]
        sql = q.get("query") or q.get("SQL") or q.get("sql")
        question = q.get("question", "<no question>")

        print("\n==============================")
        print(f"Query #{i+41}")
        print(f"DB: {db_id}")
        print(f"Question: {question}")
        print(f"Gold SQL: {sql}")
        print("==============================\n")

        schema_entry = find_schema_entry(tables_meta, db_id)
        if not schema_entry:
            print(f"  ❌ No schema found for DB {db_id}, skipping.")
            continue

        db_output_dir = OUTPUT_DIR / db_id
        generate_tests_for_query(
            db_id=db_id,
            query_index=i+40,
            gold_query=sql,
            schema_map=build_schema_map(schema_entry),
            output_dir=db_output_dir,
        )


if __name__ == "__main__":
    run_all_queries(limit=10)
