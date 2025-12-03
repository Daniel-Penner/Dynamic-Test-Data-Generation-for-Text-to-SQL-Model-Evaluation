# scripts/generate_tests_for_query.py

from pathlib import Path
import duckdb
import pandas as pd
import re

from scripts.build_tests.unit_test import SQLUnitTest
from scripts.build_tests.extract_sql_constraints import parse_constraints
from scripts.build_tests.synthetic_data_generator import generate_synthetic_dataset
from scripts.build_tests.scenario_definitions import build_scenarios
from scripts.build_tests.parse_order_limit import parse_order_limit

def _rewrite_mysql_limit(sql):
    # Matches: LIMIT 9, 2   OR LIMIT 9 ,2  etc.
    m = re.search(r"LIMIT\s+(\d+)\s*,\s*(\d+)", sql, flags=re.IGNORECASE)
    if not m:
        return sql
    offset, count = m.group(1), m.group(2)
    return re.sub(r"LIMIT\s+\d+\s*,\s*\d+",
                  f"LIMIT {count} OFFSET {offset}",
                  sql,
                  flags=re.IGNORECASE)

def _fix_strftime(sql):
    # Rewrites: strftime('%Y', OpenDate) → strftime(OpenDate, '%Y')
    return re.sub(
        r"strftime\(\s*'([^']+)'\s*,\s*([A-Za-z0-9_]+)\s*\)",
        r"strftime(\2, '\1')",
        sql
    )

def generate_tests_for_query(
    db_id: str,
    query_index: int,
    gold_query: str,
    schema_map,
    output_dir: Path,
    n_rows_per_table=12,
):

    constraints = parse_constraints(gold_query, schema_map)
    order_info = parse_order_limit(gold_query)

    scenarios = build_scenarios(
        sql=gold_query,
        limit=order_info["limit"] if order_info else None
    )
    saved = 0

    output_dir.mkdir(parents=True, exist_ok=True)

    for sc in scenarios:
        dataset = generate_synthetic_dataset(
            schema_map=schema_map,
            constraints=constraints,
            sql=gold_query,
            order_info=order_info,
            n_rows_per_table=n_rows_per_table,
            scenario=sc,
        )


        expected = run_gold_on_data(dataset, gold_query)
        if expected is None:
            continue

        test_name = sc["name"]
        filename = f"test_{query_index:03d}_{test_name}.json"
        path = output_dir / filename

        SQLUnitTest(
            db_id=db_id,
            query_index=query_index,
            scenario=sc["name"],
            sql=gold_query,
            tables=dataset,
            expected_output=expected,
        ).to_json(path)

        saved += 1

    if saved == 0:
        print(f"⚠ No tests produced for query #{query_index}")


def run_gold_on_data(dataset: dict, gold_sql: str):
    """
    Execute a gold SQL query on the synthetic dataset.
    Returns a pandas DataFrame or None.
    """

    # Pre-rewrites required for DuckDB compatibility
    gold_sql = _rewrite_mysql_limit(gold_sql)
    gold_sql = _fix_strftime(gold_sql)

    # Create connection
    con = duckdb.connect()

    # Register base tables
    for table_name, df in dataset.items():
        try:
            con.register(table_name.lower(), df)
        except Exception as e:
            print(f"[run_gold_on_data] Failed to register table {table_name}: {e}")
            con.close()
            return None

    # Replace backticks with double quotes
    sql_clean = gold_sql.replace("`", '"')

    # ============================================================
    # Register aliases (T1, T2) so joins work the same way as real DBs
    # ============================================================
    alias_pattern = r'\b([a-zA-Z0-9_]+)\s+(?:AS\s+)?([a-zA-Z0-9_]+)\b'
    for base, alias in re.findall(alias_pattern, sql_clean, flags=re.IGNORECASE):
        base_l = base.lower()
        alias_l = alias.lower()

        if base_l in dataset and alias_l not in dataset:
            try:
                con.register(alias_l, dataset[base_l])
            except Exception as e:
                print(f"[run_gold_on_data] Failed to register alias {alias_l} for base {base_l}: {e}")

    # ============================================================
    # Execute
    # ============================================================
    try:
        result = con.execute(sql_clean).df()
        con.close()
        return result

    except Exception as e:
        print("[run_gold_on_data] ERROR executing SQL:\n", e)
        con.close()
        return None