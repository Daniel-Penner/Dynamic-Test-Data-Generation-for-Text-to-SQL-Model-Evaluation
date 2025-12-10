# scripts/generate_tests_for_query.py

from pathlib import Path
import duckdb
import pandas as pd
import re

from scripts.build_tests.unit_test import SQLUnitTest
from scripts.build_tests.extract_sql_constraints import parse_constraints
from scripts.build_tests.synthetic_data_generator_BIRD import generate_synthetic_dataset
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

def _fix_strftime(sql: str) -> str:
    """
    Rewrite SQLite-style strftime('%Y', col) into DuckDB-style:
        strftime(CAST(col AS DATE), '%Y')
    Also fix the reversed argument order (DuckDB requires (value, format)).
    """

    import re

    # Pattern matching: strftime('%Y', OpenDate)
    pattern = r"strftime\(\s*'([^']+)'\s*,\s*([a-zA-Z0-9_\.]+)\s*\)"

    def repl(match):
        fmt = match.group(1)
        col = match.group(2)
        # DuckDB form: strftime(CAST(col AS DATE), 'fmt')
        return f"strftime(CAST({col} AS DATE), '{fmt}')"

    sql = re.sub(pattern, repl, sql)

    # Pattern matching reversed arguments that DuckDB errors on:
    #   strftime(OpenDate, '%Y')
    pattern2 = r"strftime\(\s*([a-zA-Z0-9_\.]+)\s*,\s*'([^']+)'\s*\)"

    def repl2(match):
        col = match.group(1)
        fmt = match.group(2)
        return f"strftime(CAST({col} AS DATE), '{fmt}')"

    return re.sub(pattern2, repl2, sql)

def generate_tests_for_query(
    db_id: str,
    query_index: int,
    gold_query: str,
    schema_map,
    output_dir: Path,
    n_rows_per_table=12,
):
    non_empty = 0
    empty = 0
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

        gold_clean = " ".join(gold_query.replace("\n", " ").split()).rstrip(";")
        expected = run_gold_on_data(dataset, gold_clean)
        if expected is None:
            continue
        
        if expected.empty:
            empty +=1
        else:
            non_empty +=1
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
    return empty, non_empty

# ------------------------------------------------------------
# LIMIT x,y → LIMIT y OFFSET x (MySQL → DuckDB)
# ------------------------------------------------------------
def rewrite_mysql_limit(sql: str) -> str:
    m = re.search(r"LIMIT\s+(\d+)\s*,\s*(\d+)", sql, flags=re.IGNORECASE)
    if not m:
        return sql
    offset, count = m.group(1), m.group(2)
    return re.sub(
        r"LIMIT\s+\d+\s*,\s*\d+",
        f"LIMIT {count} OFFSET {offset}",
        sql,
        flags=re.IGNORECASE,
    )


# ------------------------------------------------------------
# STRFTIME normalization (SQLite → DuckDB)
# ------------------------------------------------------------
def normalize_strftime(sql: str) -> str:
    # strftime('%Y', col) → strftime(CAST(col AS DATE), '%Y')
    sql = re.sub(
        r"strftime\(\s*'([^']+)'\s*,\s*([a-zA-Z0-9_.]+)\s*\)",
        r"strftime(CAST(\2 AS DATE), '\1')",
        sql,
        flags=re.IGNORECASE,
    )

    # reversed args
    sql = re.sub(
        r"strftime\(\s*([a-zA-Z0-9_.]+)\s*,\s*'([^']+)'\s*\)",
        r"strftime(CAST(\1 AS DATE), '\2')",
        sql,
        flags=re.IGNORECASE,
    )

    return sql


# ------------------------------------------------------------
# IIF → CASE WHEN
# ------------------------------------------------------------
def normalize_iif(sql: str) -> str:
    return re.sub(
        r"IIF\s*\(([^,]+),([^,]+),([^)]+)\)",
        r"(CASE WHEN \1 THEN \2 ELSE \3 END)",
        sql,
        flags=re.IGNORECASE,
    )


# ------------------------------------------------------------
# DATE LIKE 'YYYY-MM%' → CAST(date AS VARCHAR)
# ------------------------------------------------------------
def normalize_date_like(sql: str) -> str:
    return re.sub(
        r"(\w+)\.(\w+)\s+LIKE\s+'(\d{4}-\d{2})%'",
        r"CAST(\1.\2 AS VARCHAR) LIKE '\3%'",
        sql,
        flags=re.IGNORECASE,
    )


def normalize_order_by_non_projected(sql: str) -> str:
    """
    Rewrite ONLY when:
      - ORDER BY bare column
      - LIMIT present
      - ORDER BY column not an expression
    Safe for DuckDB. Conservative by design.
    """

    sql_upper = sql.upper()
    if "ORDER BY" not in sql_upper or "LIMIT" not in sql_upper:
        return sql

    # Extract ORDER BY <expr> [ASC|DESC]
    m_order = re.search(
        r"ORDER\s+BY\s+(?P<expr>[^,]+?)\s*(?P<dir>ASC|DESC)?\s+LIMIT",
        sql,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if not m_order:
        return sql

    order_expr = m_order.group("expr").strip()
    order_dir = (m_order.group("dir") or "").upper()

    # 🚫 Do NOT touch expressions
    if any(tok in order_expr for tok in ["(", ")", "/", "*", "+", "-", "CAST", " AS "]):
        return sql

    # Must be bare column or table.column
    if not re.fullmatch(r"[a-zA-Z_][a-zA-Z0-9_]*(\.[a-zA-Z_][a-zA-Z0-9_]*)?", order_expr):
        return sql

    # Extract main SELECT / FROM / LIMIT
    m = re.search(
        r"""
        SELECT\s+(?P<select>.+?)
        \s+FROM\s+(?P<from>.+?)
        \s+ORDER\s+BY\s+(?P<order>.+?)
        \s+LIMIT\s+(?P<limit>\d+)
        """,
        sql,
        flags=re.IGNORECASE | re.DOTALL | re.VERBOSE,
    )
    if not m:
        return sql

    select = m.group("select").strip()
    from_ = m.group("from").strip()
    limit = m.group("limit").strip()

    # Remove table qualifiers from outer projection
    outer_select = re.sub(r"\b\w+\.", "", select)

    return f"""
    SELECT {outer_select}
    FROM (
        SELECT {select}, {order_expr} AS __order_col
        FROM {from_}
    )
    ORDER BY __order_col {order_dir}
    LIMIT {limit}
    """.strip()

# ------------------------------------------------------------
# GROUP BY strictness → ANY_VALUE (LAST)
# ------------------------------------------------------------
def normalize_group_by(sql: str) -> str:
    if "GROUP BY" not in sql.upper():
        return sql

    m = re.search(r"SELECT\s+(.*?)\s+FROM", sql, re.I | re.S)
    if not m:
        return sql

    select_clause = m.group(1)
    parts = [p.strip() for p in select_clause.split(",")]

    new_parts = []
    for p in parts:
        if any(fn in p.upper() for fn in ("COUNT(", "SUM(", "AVG(", "MIN(", "MAX(")):
            new_parts.append(p)
        else:
            new_parts.append(f"ANY_VALUE({p})")

    new_select = ", ".join(new_parts)

    # Only replace the SELECT list, not other occurrences of the same text
    start, end = m.span(1)
    return sql[:start] + new_select + sql[end:]

def normalize_mixed_aggregates(sql: str) -> str:
    """
    If SELECT contains BOTH aggregated and non-aggregated columns
    without GROUP BY, wrap non-aggregates in ANY_VALUE().
    """
    upper = sql.upper()
    if "GROUP BY" in upper:
        return sql

    if not any(fn in upper for fn in ("AVG(", "MAX(", "MIN(", "SUM(", "COUNT(")):
        return sql

    # Find SELECT ... FROM
    m = re.search(
        r"SELECT\s+(?P<select>.+?)\s+FROM",
        sql,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if not m:
        return sql

    select_clause = m.group("select")

    parts = [p.strip() for p in select_clause.split(",")]
    new_parts = []

    for p in parts:
        if any(fn in p.upper() for fn in ("AVG(", "MAX(", "MIN(", "SUM(", "COUNT(")):
            new_parts.append(p)
        else:
            new_parts.append(f"ANY_VALUE({p})")

    return sql.replace(select_clause, ", ".join(new_parts))

def normalize_double_quoted_strings(sql: str) -> str:
    """
    SQLite allows double-quoted string literals.
    DuckDB treats them as identifiers.
    Rewrites:
        col = "text"  →  col = 'text'
    """

    return re.sub(
        r'=\s*"([^"]+)"',
        r"= '\1'",
        sql
    )


# ------------------------------------------------------------
# ✅ MASTER NORMALIZER (USE THIS ONLY)
# ------------------------------------------------------------
def normalize_gold_sql(sql: str, debug: bool = False) -> str:
    original = sql

    pipeline = [
        ("normalize_double_quoted_strings", normalize_double_quoted_strings),
        ("rewrite_mysql_limit", rewrite_mysql_limit),
        ("normalize_strftime", normalize_strftime),
        ("normalize_iif", normalize_iif),
        ("normalize_date_like", normalize_date_like),
        ("normalize_order_by_non_projected", normalize_order_by_non_projected),
        ("normalize_mixed_aggregates", normalize_mixed_aggregates),
        ("normalize_group_by", normalize_group_by),  # MUST be last
    ]

    for name, fn in pipeline:
        new_sql = fn(sql)
        if debug and new_sql != sql:
            print(f"[SQL normalize] {name} applied")
        sql = new_sql

    return sql


def run_gold_on_data(dataset: dict, gold_sql: str):
    """
    Execute a gold SQL query on the synthetic dataset.
    Returns a pandas DataFrame or None.
    """

    # Pre-rewrites required for DuckDB compatibility
    gold_sql = normalize_gold_sql(gold_sql)

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

    sql_clean = re.sub(
        r"\(\s*select\s+([^)]+?)\s+order\s+by\s+([^)]+?)\s+limit\s+1\s*\)",
        r"(select \1 order by \2 limit 1)",
        sql_clean,
        flags=re.IGNORECASE | re.DOTALL
    )

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