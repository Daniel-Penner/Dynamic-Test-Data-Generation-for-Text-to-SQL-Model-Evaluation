import sqlite3
import duckdb
import pandas as pd
from pathlib import Path
import os
import re

OUT_DIR = Path("bird_input_data/duckdb_databases_fixed")
OUT_DIR.mkdir(exist_ok=True, parents=True)

RESERVED = {"order", "match", "group", "table", "select"}

SQL_RESERVED = {
    "select","from","where","join","inner","left","right","full","outer",
    "on","group","order","by","having","limit","offset","insert","update",
    "delete","into","values","table","create","drop","union","except",
    "intersect","case","when","then","else","end","and","or","not","as",
    "match","cross"
}

IDENTIFIER_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")

def fix_ident(name: str) -> str:
    """
    Quote identifiers if:
      - they are SQL keywords
      - contain non-word characters
      - start with digits
      - are mixed-case
    """
    name_lower = name.lower()

    # must quote if keyword
    if name_lower in SQL_RESERVED:
        return f'"{name}"'

    # must quote if not a simple identifier
    if not IDENTIFIER_RE.match(name):
        return f'"{name}"'

    # duckdb is case-insensitive unless quoted
    if not name.islower():
        return f'"{name}"'

    return name


def map_sqlite_type(t: str) -> str:
    """
    Map SQLite's loose type system → DuckDB types.
    """
    if t is None:
        return "TEXT"

    t_upper = t.upper()

    if "INT" in t_upper:
        return "INTEGER"
    if "CHAR" in t_upper or "TEXT" in t_upper or "CLOB" in t_upper:
        return "TEXT"
    if "REAL" in t_upper or "FLOA" in t_upper or "DOUB" in t_upper:
        return "DOUBLE"
    if "NUM" in t_upper or "DEC" in t_upper:
        return "DOUBLE"
    if "DATE" in t_upper or "TIME" in t_upper:
        return "DATE"

    # fallback
    return "TEXT"


def convert_one(sqlite_path: Path):
    print(f"\nConverting {sqlite_path} → {OUT_DIR/(sqlite_path.stem+'.duckdb')}")

    conn_sql = sqlite3.connect(sqlite_path)
    duck_path = OUT_DIR / f"{sqlite_path.stem}.duckdb"

    if duck_path.exists():
        duck_path.unlink()

    conn_duck = duckdb.connect(str(duck_path))

    # list tables
    tables = pd.read_sql_query(
        "SELECT name FROM sqlite_master WHERE type='table';",
        conn_sql
    )["name"].tolist()

    for tbl in tables:
        print(f"  → Loading table {tbl}")

        # get schema
        pragma = pd.read_sql_query(f"PRAGMA table_info('{tbl}')", conn_sql)

        cols = pragma["name"].tolist()
        types = pragma["type"].tolist()

        # build CREATE TABLE
        col_defs = []
        for col, typ in zip(cols, types):
            duck_type = map_sqlite_type(typ)
            col_defs.append(f"{fix_ident(col)} {duck_type}")

        create_sql = f"CREATE TABLE {fix_ident(tbl)} ({', '.join(col_defs)});"
        conn_duck.execute(create_sql)

        # load all rows
        df = pd.read_sql_query(f"SELECT * FROM '{tbl}'", conn_sql)

        # insert
        conn_duck.register("tmp_df", df)
        conn_duck.execute(f"INSERT INTO {fix_ident(tbl)} SELECT * FROM tmp_df;")
        conn_duck.unregister("tmp_df")

    conn_sql.close()
    conn_duck.close()


def main():
    root = Path("bird_input_data/dev_databases")
    sqlite_files = list(root.rglob("*.sqlite"))

    if not sqlite_files:
        print("No SQLite DBs found.")
        return

    for f in sqlite_files:
        convert_one(f)


if __name__ == "__main__":
    main()
