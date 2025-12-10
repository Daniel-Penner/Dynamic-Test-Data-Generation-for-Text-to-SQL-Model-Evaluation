from __future__ import annotations

import sys
from pathlib import Path

from scripts.build_tests.generate_tests_Spider import generate_tests_for_query
from scripts.build_tests.build_schema_map import build_schema_map

PROJECT_ROOT = Path(__file__).resolve().parents[2]
while PROJECT_ROOT.name != "Dynamic-Test-Data-Generation-for-Text-to-SQL-Model-Evaluation":
    PROJECT_ROOT = PROJECT_ROOT.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

SPIDER_PATH = PROJECT_ROOT / "spider_input_data"
SPIDER_DIR = SPIDER_PATH / "database"
GOLD_PATH = SPIDER_PATH / "dev_gold.sql"
SCHEMA_META = SPIDER_PATH / "tables.json"   # standard Spider schema metadata
OUTPUT_DIR = PROJECT_ROOT / "tests_SPIDER"


def load_spider_gold():
    pairs = []
    with open(GOLD_PATH, "r", encoding="utf8") as f:
        for i, line in enumerate(f):
            sql, db_id = line.strip().split("\t")
            pairs.append((i, sql, db_id))
    return pairs


def find_schema_entry(tables, db_id):
    for t in tables:
        if t["db_id"] == db_id:
            return t
    return None


def run_all_queries(start=0, limit=None):
    gold = load_spider_gold()

    import json
    with open(SCHEMA_META, "r", encoding="utf8") as f:
        tables_meta = json.load(f)

    end = start + limit if limit else len(gold)

    for idx, sql, db_id in gold[start:end]:
        print("\n==============================")
        print(f"Spider Query #{idx}")
        print(f"DB: {db_id}")
        print(f"Gold SQL: {sql}")
        print("==============================")

        schema = find_schema_entry(tables_meta, db_id)
        if not schema:
            print(f"❌ No schema found for {db_id}")
            continue

        out_dir = OUTPUT_DIR / db_id

        empty, non_empty = generate_tests_for_query(
            db_id=db_id,
            query_index=idx,
            gold_query=sql,
            schema_map=build_schema_map(schema),
            output_dir=out_dir
        )

        total = empty + non_empty
        print(f"✅ valid tests: {non_empty}/{total}")
        print(f"❌ empty tests: {empty}/{total}")


if __name__ == "__main__":
    run_all_queries(start=0, limit=None)
