# scripts/generate_dynamic_data.py
import json
import sqlite3
from pathlib import Path
import sqlparse
import pandas as pd
import duckdb
import random
import os
from openai import OpenAI
from dotenv import load_dotenv

# -------------------------------------------------------------------
# ENV + PATHS
# -------------------------------------------------------------------
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

PROJECT_ROOT = Path(__file__).resolve().parents[1]
BIRD_PATH = PROJECT_ROOT / "bird_input_data"
DBS_PATH = BIRD_PATH / "dev_databases"

QUERIES_PATH = BIRD_PATH / "dev.json"
TABLES_PATH = BIRD_PATH / "dev_tables.json"

# -------------------------------------------------------------------
# LOAD DATA
# -------------------------------------------------------------------
with open(QUERIES_PATH, "r", encoding="utf-8") as f:
    queries = json.load(f)
with open(TABLES_PATH, "r", encoding="utf-8") as f:
    tables = json.load(f)

first_query = queries[0]
db_id = first_query["db_id"]
gold_query = (
    first_query.get("query")
    or first_query.get("SQL")
    or first_query.get("SQL_query")
    or first_query.get("sql")
)
question = first_query.get("question", "<no question text found>")
print("Natural language question:", question)
print("SQL query:", gold_query)
print("Database ID:", db_id)

db_file = DBS_PATH / db_id / f"{db_id}.sqlite"
if not db_file.exists():
    raise FileNotFoundError(f"❌ Could not find database file at: {db_file}")

conn = sqlite3.connect(db_file)
cursor = conn.cursor()
cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
tables_in_db = [t[0] for t in cursor.fetchall()]
print("\nTables in DB:", tables_in_db)
conn.close()

# -------------------------------------------------------------------
# UTILS
# -------------------------------------------------------------------
def extract_query_info(sql_query: str):
    parsed = sqlparse.parse(sql_query)[0]
    tokens = [t for t in parsed.tokens if not t.is_whitespace]
    cols, consts, ops = [], [], []
    for t in tokens:
        t_str = t.value
        if "`" in t_str:
            cols.append(t_str.replace("`", ""))
        if "'" in t_str:
            consts.append(t_str.strip("'"))
        if any(o in t_str for o in ["=", ">", "<", "LIKE", "IN", "/"]):
            ops.append(t_str)
    return {"columns": list(set(cols)), "constants": consts, "operators": ops}

def evaluate_gold_query(df: pd.DataFrame, sql_query: str):
    con = duckdb.connect()
    con.register("frpm", df)
    try:
        result = con.execute(sql_query).df()
    except Exception as e:
        print(f"⚠️ Query execution error: {e}")
        result = None
    con.close()
    return result

# -------------------------------------------------------------------
# LLM-BASED DATA GENERATOR
# -------------------------------------------------------------------
def generate_rows_with_llm(schema_snippet: str, sql_query: str, n_rows: int = 6):
    prompt = f"""
You are creating synthetic SQL test data.
Given the table schema and gold SQL query, generate {n_rows} rows of JSON data
that would test this query's logic, including normal and edge cases.

Schema:
{schema_snippet}

Gold SQL query:
{sql_query}

Rules:
- Return ONLY valid JSON list of objects (no markdown, no explanation).
- Each object should include every column that appears in the query or WHERE clauses.
- Include at least one row that triggers an edge condition (e.g., zero division, filter mismatch).
"""
    completion = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.4,
    )
    content = completion.choices[0].message.content.strip()
    # clean up and parse
    try:
        if content.startswith("```"):
            content = content.split("```")[1]
            if content.startswith("json"):
                content = content[len("json"):].strip()
        data = json.loads(content)
        return pd.DataFrame(data)
    except Exception as e:
        print(f"⚠️ Failed to parse LLM output: {e}\nRaw content:\n{content}")
        return pd.DataFrame()

# -------------------------------------------------------------------
# PIPELINE
# -------------------------------------------------------------------
info = extract_query_info(gold_query)
print("\nExtracted query info:", info)

schema_entry = next((t for t in tables if t["db_id"] == db_id), None)

if not schema_entry:
    raise ValueError(f"No schema found for database ID: {db_id}")

# Get the table names and columns for the relevant db
table_names = schema_entry["table_names_original"]
columns = schema_entry["column_names_original"]
column_types = schema_entry["column_types"]

# Create a structured mapping: {table_name: [(col_name, col_type), ...]}
schema_map = {name: [] for name in table_names}
for (table_id, col_name), col_type in zip(columns, column_types):
    if table_id != -1:
        schema_map[table_names[table_id]].append((col_name, col_type))

# Pick only the tables actually mentioned in the query
query_tables = [
    t for t in table_names if any(t.lower() in gold_query.lower() for t in [t, f"`{t}`"])
]
if not query_tables:
    query_tables = table_names  # fallback if nothing matched

schema_snippet = json.dumps(
    {t: schema_map[t] for t in query_tables},
    indent=2
)
print("\nSchema snippet used for LLM generation:")
print(schema_snippet)

print("\nRequesting LLM-generated data...")
df_generated = generate_rows_with_llm(schema_snippet, gold_query)
print("\nGenerated synthetic data (LLM):")
print(df_generated)

if not df_generated.empty:
    result = evaluate_gold_query(df_generated, gold_query)
    print("\nGold query output on generated data:")
    print(result)
else:
    print("\nNo generated data to evaluate.")
