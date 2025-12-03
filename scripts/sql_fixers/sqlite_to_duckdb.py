import sqlparse
import re

# ------------------------------------------------------
# LIMIT offset,count → LIMIT count OFFSET offset
# (SQLite/MySQL style → DuckDB style)
# ------------------------------------------------------
def fix_limit(sql: str) -> str:
    if not sql:
        return sql

    # Matches LIMIT x, y
    return re.sub(
        r"LIMIT\s+(\d+)\s*,\s*(\d+)",
        r"LIMIT \2 OFFSET \1",
        sql,
        flags=re.IGNORECASE,
    )


# ------------------------------------------------------
# Fix strftime('%Y', col) → DuckDB-compatible format
# DuckDB allows strftime(col, '%Y') directly
# ------------------------------------------------------
def fix_strftime(sql: str) -> str:
    if not sql:
        return sql

    # Matches: strftime('%Y', col)
    return re.sub(
        r"strftime\s*\(\s*'(%Y|%m|%d)'\s*,\s*([A-Za-z0-9_\.\"']+)\s*\)",
        lambda m: f"strftime({m.group(2)}, '{m.group(1)}')",
        sql,
        flags=re.IGNORECASE,
    )


# ------------------------------------------------------
# Replace MySQL-style backticks (`col`) with DuckDB double quotes
# ------------------------------------------------------
def fix_quotes(sql: str) -> str:
    if not sql:
        return sql
    return sql.replace("`", '"')


# ------------------------------------------------------
# Main SQL normalization for DuckDB
# ------------------------------------------------------
def canonicalize_sql(sql: str) -> str:
    if sql is None:
        return sql

    sql = sql.strip()

    # independent transformations
    sql = fix_quotes(sql)
    sql = fix_limit(sql)
    sql = fix_strftime(sql)

    return sql
