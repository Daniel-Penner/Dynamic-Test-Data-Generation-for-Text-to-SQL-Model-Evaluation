# scripts/generate_dynamic_data.py
import json
from pathlib import Path
import sqlparse
import pandas as pd
import duckdb
import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

PROJECT_ROOT = Path(__file__).resolve().parents[1]
BIRD_PATH = PROJECT_ROOT / "bird_input_data"

QUERIES_PATH = BIRD_PATH / "dev.json"
TABLES_PATH = BIRD_PATH / "dev_tables.json"

with open(QUERIES_PATH, "r", encoding="utf-8") as f:
    queries = json.load(f)

with open(TABLES_PATH, "r", encoding="utf-8") as f:
    tables = json.load(f)

#Just first five queries for now
for i, query_item in enumerate(queries[:5], start=1):
    db_id = query_item["db_id"]
    gold_query = (
        query_item.get("query")
        or query_item.get("SQL")
        or query_item.get("SQL_query")
        or query_item.get("sql")
    )
    question = query_item.get("question", "<no question text found>")
    
    print(f"\n--- Query {i} ---")
    print("Natural language question:", question)
    print("SQL query:", gold_query)
    print("Database ID:", db_id)

# Use the first one for downstream testing
first_query = queries[0]
db_id = first_query["db_id"]
gold_query = (
    first_query.get("query")
    or first_query.get("SQL")
    or first_query.get("SQL_query")
    or first_query.get("sql")
)
question = first_query.get("question", "<no question text found>")

#Schema extraction/formatting
schema_entry = next((t for t in tables if t["db_id"] == db_id), None)
if not schema_entry:
    raise ValueError(f"No schema found for database ID: {db_id}")

table_names = schema_entry["table_names_original"]
columns = schema_entry["column_names_original"]
column_types = schema_entry["column_types"]

schema_map = {name: [] for name in table_names}
for (table_id, col_name), col_type in zip(columns, column_types):
    if table_id != -1:
        schema_map[table_names[table_id]].append((col_name, col_type))

query_tables = [
    t for t in table_names if any(t.lower() in gold_query.lower() for t in [t, f"`{t}`"])
]
if not query_tables:
    query_tables = table_names

schema_snippet = json.dumps({t: schema_map[t] for t in query_tables}, indent=2)

def extract_query_info(sql_query: str):
    """Extracts columns, constants, and operators from SQL."""
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

def coerce_df_to_schema(df: pd.DataFrame, schema_map: dict, table_name: str) -> pd.DataFrame:
    """Coerce columns in df to types declared in dev_tables.json schema_map."""
    TYPE_MAP = {
        "integer": "Int64",
        "real": "float64",
        "text": "string",
    }
    df = df.copy()
    for col_name, col_type in schema_map.get(table_name, []):
        if col_name in df.columns:
            target = TYPE_MAP.get(col_type, None)
            if target == "float64":
                df[col_name] = pd.to_numeric(df[col_name], errors="coerce")
            elif target == "Int64":
                df[col_name] = pd.to_numeric(df[col_name], errors="coerce").astype("Int64")
            elif target == "string":
                df[col_name] = df[col_name].astype("string")
    return df

info = extract_query_info(gold_query)
print("\nExtracted query info:", info)

#Generate Data
def generate_rows_with_llm(schema_snippet: str, sql_query: str, n_rows: int = 6):
    prompt = f"""
You are generating synthetic SQL test data for benchmark evaluation.

Goal: create {n_rows} JSON rows of data such that when the following gold SQL query
is executed, it produces a valid, non-empty, non-null result set.

Schema:
{schema_snippet}

Gold SQL query:
{sql_query}

Requirements:
- The data MUST cause the SQL query to return at least one valid numeric result (no NULLs).
- All numeric divisions must have non-zero denominators.
- Include a mix of rows: some that satisfy the WHERE clause and some that do not.
- Use realistic value ranges consistent with the column types.
- Return ONLY a JSON array (no markdown, no commentary).
"""

    completion = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.4,
    )
    content = completion.choices[0].message.content.strip()

    # Clean and parse JSON safely
    try:
        if content.startswith("```"):
            content = content.split("```")[1]
            if content.startswith("json"):
                content = content[len("json"):].strip()
        data = json.loads(content)
        df = pd.DataFrame(data)
        return df
    except Exception as e:
        print(f"Failed to parse LLM output: {e}\nRaw output:\n{content}")
        return pd.DataFrame()


print("\nRequesting LLM-generated synthetic data...")
df_generated = generate_rows_with_llm(schema_snippet, gold_query)

print("\nGenerated synthetic data:")
print(df_generated)

#Evaluate gold query on data
def evaluate_gold_query(df: pd.DataFrame, sql_query: str, table_name: str, schema_map: dict):
    """Run the gold query on synthetic data using DuckDB."""
    df = coerce_df_to_schema(df, schema_map, table_name)

    duckdb_query = sql_query.replace("`", "\"")

    cols_needed = ["County Name", "Free Meal Count (K-12)", "Enrollment (K-12)"]
    present = [c for c in cols_needed if c in df.columns]
    if present:
        print("\n[DEBUG] Alameda rows (relevant columns):")
        print(df.loc[df["County Name"] == "Alameda", present])
        print("\n[DEBUG] dtypes of relevant columns:")
        print(df[present].dtypes)

    import duckdb
    con = duckdb.connect()
    con.register(table_name, df)
    try:
        result = con.execute(duckdb_query).df()
    except Exception as e:
        print(f"Query execution error: {e}")
        result = None
    finally:
        con.close()
    return result



if not df_generated.empty:
    result = evaluate_gold_query(df_generated, gold_query, query_tables[0], schema_map)
    print("\nGold query output on synthetic data:")
    print(result)
else:
    print("\nNo generated data to evaluate.")


# -------------------------------------------------------------------
# ADDED SECTION: Self-validating dynamic generation loop
# -------------------------------------------------------------------

def run_self_validating_generation(schema_snippet, gold_query, table_name, schema_map, max_retries=3):
    """
    Repeatedly generate data and test it until the gold SQL query yields
    a non-empty, non-null, valid result. Ensures gold query always produces output.
    """
    for attempt in range(max_retries):
        print(f"\n🔁 Attempt {attempt+1} to generate valid data...")
        df = generate_rows_with_llm(schema_snippet, gold_query)
        if df.empty:
            print("⚠️ LLM returned empty or unparsable output. Retrying...")
            continue

        df = coerce_df_to_schema(df, schema_map, table_name)
        duckdb_query = gold_query.replace("`", "\"")

        con = duckdb.connect()
        con.register(table_name, df)
        try:
            result = con.execute(duckdb_query).df()
        except Exception as e:
            print(f"⚠️ Query error: {e}")
            con.close()
            continue
        con.close()

        if result is not None and not result.empty and not result.isna().any().any():
            print("✅ Valid non-null result obtained.")
            return df, result
        else:
            print("⚠️ Gold query produced null or empty result. Regenerating...")
    raise RuntimeError("❌ Failed to generate valid dynamic data after multiple attempts.")


# -------------------------------------------------------------------
# ADDED SECTION: Optional automatic retry pipeline
# -------------------------------------------------------------------

def auto_generate_and_validate(schema_snippet, gold_query, query_tables, schema_map):
    """
    Wrapper pipeline that uses self-validation to guarantee gold SQL success.
    """
    try:
        df_generated, result = run_self_validating_generation(
            schema_snippet, gold_query, query_tables[0], schema_map
        )
        print("\nFinal gold query output on synthetic data:")
        print(result)
        return df_generated, result
    except RuntimeError as e:
        print(e)
        return None, None


# -------------------------------------------------------------------
# ADDED SECTION: Run self-validation automatically for first 5 queries
# -------------------------------------------------------------------
print("\nRunning self-validation to ensure gold query success for first 5 queries...")

for i, query_item in enumerate(queries[:5], start=1):
    db_id = query_item["db_id"]
    gold_query = (
        query_item.get("query")
        or query_item.get("SQL")
        or query_item.get("SQL_query")
        or query_item.get("sql")
    )
    question = query_item.get("question", "<no question text found>")

    print(f"\n==========================")
    print(f"🔎 Processing Query {i}")
    print("==========================")
    print("Natural language question:", question)
    print("SQL query:", gold_query)
    print("Database ID:", db_id)

    # Extract schema for this db_id
    schema_entry = next((t for t in tables if t["db_id"] == db_id), None)
    if not schema_entry:
        print(f"❌ No schema found for database ID: {db_id}")
        continue

    table_names = schema_entry["table_names_original"]
    columns = schema_entry["column_names_original"]
    column_types = schema_entry["column_types"]

    schema_map = {name: [] for name in table_names}
    for (table_id, col_name), col_type in zip(columns, column_types):
        if table_id != -1:
            schema_map[table_names[table_id]].append((col_name, col_type))

    query_tables = [
        t for t in table_names if any(t.lower() in gold_query.lower() for t in [t, f"`{t}`"])
    ]
    if not query_tables:
        query_tables = table_names

    schema_snippet = json.dumps({t: schema_map[t] for t in query_tables}, indent=2)

    # Run self-validating generation pipeline for this query
    df_generated, result = auto_generate_and_validate(schema_snippet, gold_query, query_tables, schema_map)

    if result is not None:
        print(f"\n✅ Query {i} completed successfully.")
        print(result)
    else:
        print(f"\n⚠️ Query {i} failed to produce a valid output after retries.")

