import argparse
import json
import sqlite3
from pathlib import Path
import multiprocessing as mp

from evaluation.evaluator_core import (
    execute_sqlite_query,
    compare_results,
)

def safe(s):
    return s.encode('unicode_escape').decode('utf8')

def load_predictions(pred_path: str):
    """
    Expected format:
    {
        "0": "SELECT ... \t----- bird -----\tdbname",
        "1": "SELECT ... \t----- bird -----\tdbname",
        ...
    }
    OR list [{"db_id": "...", "pred": "..."}]
    """
    with open(pred_path, "r", encoding="utf8") as f:
        data = json.load(f)

    preds = []
    if isinstance(data, dict):
        # BIRD DeepSeek format
        for _, row in data.items():
            try:
                sql, dbname = row.split("\t----- bird -----\t")
                preds.append({"db_id": dbname.strip(), "pred": sql.strip()})
            except ValueError:
                continue
    elif isinstance(data, list):
        preds = data
    else:
        raise ValueError("Unsupported predictions format.")

    return preds


def load_bird_gold(gold_path: str):
    """
    Expected format: BIRD dev.json
    [
        {
            "query": "...",
            "db_id": "california_schools",
            ...
        }
    ]
    """
    with open(gold_path, "r", encoding="utf8") as f:
        data = json.load(f)

    gold = []
    for q in data:
        sql = (
            q.get("query")
            or q.get("SQL")
            or q.get("gold")
            or q.get("gold_query")
        )
        if not sql:
            raise ValueError(f"No gold SQL found in record: {q}")

        gold.append({
            "db_id": q["db_id"],
            "gold": sql.strip()
        })
    return gold


def run_bird_eval(pred_path, bird_gold_path):
    preds = load_predictions(pred_path)
    gold = load_bird_gold(bird_gold_path)

    if len(preds) != len(gold):
        print(f"[WARNING] Pred count = {len(preds)}, Gold count = {len(gold)}")
        print("Evaluation continues but EX will be misaligned.\n")

    results = []
    n = min(len(preds), len(gold))

    for i in range(n):
        p = preds[i]
        g = gold[i]

        db_name = g["db_id"]
        db_path = Path("bird_input_data/dev_databases") / f"{db_name}" / f"{db_name}.sqlite"

        if not db_path.exists():
            print(f"[ERROR] Missing DB: {db_path}")
            results.append(0)
            continue

        # Accept several possible field names
        if "pred" in p:
            pred_sql = p["pred"]
        elif "sql" in p:
            pred_sql = p["sql"]
        elif "prediction" in p:
            pred_sql = p["prediction"]
        else:
            raise KeyError(f"No SQL prediction field found in entry: {p}")
        gold_sql = g["gold"]

        try:
            pred_res = execute_sqlite_query(str(db_path), pred_sql)
        except Exception as e:
            pred_res = f"__error__:{e}"

        try:
            gold_res = execute_sqlite_query(str(db_path), gold_sql)
        except Exception as e:
            gold_res = f"__error__:{e}"

        ex_val, f1_val, ves_val = compare_results(pred_res, gold_res)
        results.append(ex_val)

        print("="*80)
        print(f"[Query {i}] DB={db_name}")
        print("GOLD SQL:", gold_sql)
        print("PRED SQL:", pred_sql)
        print("\n--- GOLD OUTPUT ---")
        print(gold_res)
        print("\n--- PRED OUTPUT ---")
        print(pred_res)
        print("\n--> EX =", ex_val)

    ex = sum(results)/len(results)
    print("\n==============================")
    print("BIRD Evaluation Summary (SQLite)")
    print("==============================")
    print("Execution Accuracy:", ex)

    return ex


# --------------------------------------------------------------------------
# ENTRYPOINT (this is the missing part you needed)
# --------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--pred", required=True,
                        help="Path to DeepSeek or model predictions JSON")

    parser.add_argument("--bird_gold", required=True,
                        help="Path to BIRD dev.json or test.json")

    args = parser.parse_args()

    run_bird_eval(
        pred_path=args.pred,
        bird_gold_path=args.bird_gold
    )
