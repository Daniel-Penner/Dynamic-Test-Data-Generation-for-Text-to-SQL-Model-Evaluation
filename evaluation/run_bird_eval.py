# run_bird_eval.py
import argparse
import csv
import json
from pathlib import Path

from evaluation.evaluator_core import (
    execute_sqlite_query,
    compare_results,
)


def safe(s):
    return s.encode("unicode_escape").decode("utf8")


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
        # BIRD DeepSeek-style format
        for _, row in sorted(data.items(), key=lambda kv: int(kv[0])):
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
            "difficulty": "...",
            ...
        }
    ]
    Returns list with index as implicit query_index.
    """
    with open(gold_path, "r", encoding="utf8") as f:
        data = json.load(f)

    gold = []
    for idx, q in enumerate(data):
        sql = (
            q.get("query")
            or q.get("SQL")
            or q.get("gold")
            or q.get("gold_query")
        )
        if not sql:
            raise ValueError(f"No gold SQL found in record: {q}")

        gold.append(
            {
                "query_index": idx,
                "db_id": q["db_id"],
                "gold": sql.strip(),
                "difficulty": q.get("difficulty"),
            }
        )
    return gold


def load_allowed_indices(id_csv: str | None):
    if not id_csv:
        return None

    allowed = set()
    with open(id_csv, "r", encoding="utf8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Expect a column named "query_index"
            allowed.add(int(row["query_index"]))
    return allowed


def run_bird_eval(pred_path, bird_gold_path, id_csv=None, out_json=None):
    preds = load_predictions(pred_path)
    gold = load_bird_gold(bird_gold_path)
    allowed = load_allowed_indices(id_csv)

    if len(preds) != len(gold):
        print(f"[WARNING] Pred count = {len(preds)}, Gold count = {len(gold)}")
        print("Evaluation continues but EX will be misaligned if ordering differs.\n")

    results = []
    per_query_records = []

    n = min(len(preds), len(gold))

    for i in range(n):
        g = gold[i]
        if allowed is not None and g["query_index"] not in allowed:
            continue  # skip queries without synthetic tests

        p = preds[i]

        db_name = g["db_id"]
        db_path = (
            Path("bird_input_data/dev_databases")
            / f"{db_name}"
            / f"{db_name}.sqlite"
        )

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
            pred_res = {"error": str(e)}

        try:
            gold_res = execute_sqlite_query(str(db_path), gold_sql)
        except Exception as e:
            gold_res = {"error": str(e)}

        ex_val, f1_val, ves_val = compare_results(pred_res, gold_res)
        results.append(ex_val)

        print("=" * 80)
        print(f"[Query {i}] DB={db_name}")
        print("GOLD SQL:", gold_sql)
        print("PRED SQL:", pred_sql)
        print("\n--- GOLD OUTPUT ---")
        print(gold_res)
        print("\n--- PRED OUTPUT ---")
        print(pred_res)
        print("\n--> EX =", ex_val)

        per_query_records.append(
            {
                "query_index": g["query_index"],
                "db_id": db_name,
                "difficulty": g.get("difficulty"),
                "gold_sql": gold_sql,
                "pred_sql": pred_sql,
                "gold_output": gold_res,
                "pred_output": pred_res,
                "gold_error": isinstance(gold_res, dict) and "error" in gold_res,
                "pred_error": isinstance(pred_res, dict) and "error" in pred_res,
                "ex": ex_val,
                "f1": f1_val,
                "ves": ves_val,
            }
        )

    ex = sum(results) / len(results) if results else 0.0
    print("\n==============================")
    print("BIRD Evaluation Summary (SQLite)")
    print("==============================")
    print("Execution Accuracy:", ex)

    # ---------------- JSON LOG ----------------
    if out_json is None:
        stem = Path(pred_path).stem
        out_json = Path(pred_path).with_name(f"bird_eval_{stem}.json")

    out_obj = {
        "meta": {
            "pred_path": str(pred_path),
            "bird_gold_path": str(bird_gold_path),
            "id_csv": str(id_csv) if id_csv else None,
            "num_queries_evaluated": len(per_query_records),
            "ex_avg": ex,
        },
        "queries": per_query_records,
    }

    with open(out_json, "w", encoding="utf8") as f:
        json.dump(out_obj, f, indent=2, ensure_ascii=False)

    print(f"\n[INFO] BIRD eval JSON written to: {out_json}")

    return ex


# --------------------------------------------------------------------------
# ENTRYPOINT
# --------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--pred", required=True,
                        help="Path to model predictions JSON")
    parser.add_argument("--bird_gold", required=True,
                        help="Path to BIRD dev.json or test.json")
    parser.add_argument("--id_csv", default=None,
                        help="Optional CSV with column query_index to filter queries.")
    parser.add_argument("--out_json", default=None,
                        help="Optional output JSON path for detailed per-query log.")

    args = parser.parse_args()

    run_bird_eval(
        pred_path=args.pred,
        bird_gold_path=args.bird_gold,
        id_csv=args.id_csv,
        out_json=args.out_json,
    )
