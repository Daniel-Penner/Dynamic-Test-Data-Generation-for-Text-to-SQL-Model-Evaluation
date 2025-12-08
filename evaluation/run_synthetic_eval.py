# run_synthetic_eval.py
import argparse
import csv
import json
import sqlite3
from pathlib import Path
from collections import defaultdict

from evaluation.evaluator_core import (
    execute_sqlite_query_conn,
    compare_results,
)


# -------------------------------------------------------------------
# Build synthetic in-memory SQLite DB
# -------------------------------------------------------------------
def build_synthetic_sqlite(tables_dict):
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()

    for table_name, rows in tables_dict.items():
        if not rows:
            continue

        cols = list(rows[0].keys())
        col_defs = ", ".join(f'"{c}" TEXT' for c in cols)
        cur.execute(f'CREATE TABLE "{table_name}" ({col_defs});')

        for row in rows:
            placeholders = ", ".join("?" for _ in cols)
            values = [row[c] for c in cols]
            cur.execute(
                f'INSERT INTO "{table_name}" VALUES ({placeholders})',
                values,
            )

    conn.commit()
    return conn


def clean_pred_sql(raw_sql: str) -> str:
    if "\t-----" in raw_sql:
        return raw_sql.split("\t-----")[0].strip()
    return raw_sql.strip()


def load_allowed_indices(id_csv: str | None):
    if not id_csv:
        return None
    allowed = set()
    with open(id_csv, "r", encoding="utf8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            allowed.add(int(row["query_index"]))
    return allowed


def load_bird_meta(bird_gold_path: str | None):
    """
    Returns dict: query_index -> {db_id, difficulty, gold_sql}
    If bird_gold_path is None, returns {}.
    """
    if not bird_gold_path:
        return {}

    with open(bird_gold_path, "r", encoding="utf8") as f:
        data = json.load(f)

    meta = {}
    for idx, q in enumerate(data):
        sql = (
            q.get("query")
            or q.get("SQL")
            or q.get("gold")
            or q.get("gold_query")
        )
        meta[idx] = {
            "db_id": q.get("db_id"),
            "difficulty": q.get("difficulty"),
            "gold_sql": sql.strip() if sql else None,
        }
    return meta


# -------------------------------------------------------------------
# Run synthetic evaluation
# -------------------------------------------------------------------
def run_synthetic_eval(tests_dir, pred_file, id_csv=None, bird_gold=None, out_json=None):
    tests_dir = Path(tests_dir)
    pred_file = Path(pred_file)

    # Load predictions dict (keys are query_index as strings)
    with open(pred_file, "r", encoding="utf8") as f:
        preds_raw = json.load(f)

    allowed = load_allowed_indices(id_csv)
    bird_meta = load_bird_meta(bird_gold)

    test_files = sorted(tests_dir.rglob("*.json"))

    # Per-test and per-query structures
    per_query_tests = defaultdict(list)

    # For strict per-query metrics
    per_query_agg = defaultdict(lambda: {
        "ex_min": 1,      # strict: if any scenario fails → ex = 0
        "f1_sum": 0.0,
        "ves_sum": 0.0,
        "count": 0,
    })

    for tf in test_files:
        with open(tf, "r", encoding="utf8") as f:
            test_obj = json.load(f)

        qid = int(test_obj["query_index"])
        if allowed is not None and qid not in allowed:
            continue

        test_name = tf.name
        tables = test_obj["tables"]
        gold_sql = test_obj["sql"]

        # Use prediction if available, else fall back to gold SQL
        qid_str = str(qid)
        if qid_str not in preds_raw:
            print(f"[WARN] Missing prediction for query_index={qid}")
            pred_sql = gold_sql
        else:
            pred_sql = clean_pred_sql(preds_raw[qid_str])

        print("=" * 80)
        print(f"[Test] {test_name} (query_index={qid})")

        # Build DB and run queries with shared evaluator_core logic
        conn = build_synthetic_sqlite(tables)

        gold_out = execute_sqlite_query_conn(conn, gold_sql)
        pred_out = execute_sqlite_query_conn(conn, pred_sql)

        ex, f1, ves = compare_results(pred_out, gold_out)

        conn.close()

        print("GOLD SQL:", gold_sql)
        print("PRED SQL:", pred_sql)
        print("\n--- GOLD OUTPUT ---")
        print(gold_out)
        print("\n--- PRED OUTPUT ---")
        print(pred_out)
        print(f"\n--> EX  = {ex}")
        print(f"--> F1  = {f1}")
        print(f"--> VES = {ves}")

        gold_error = isinstance(gold_out, dict) and "error" in gold_out
        pred_error = isinstance(pred_out, dict) and "error" in pred_out

        # Store per-test record
        per_query_tests[qid].append(
            {
                "test_file": test_name,
                "scenario_name": test_obj.get("name"),
                "gold_output": gold_out,
                "pred_output": pred_out,
                "gold_error": gold_error,
                "pred_error": pred_error,
                "ex": ex,
                "f1": f1,
                "ves": ves,
            }
        )

        # Update strict per-query aggregation
        agg = per_query_agg[qid]
        agg["ex_min"] = min(agg["ex_min"], ex)
        agg["f1_sum"] += f1
        agg["ves_sum"] += ves
        agg["count"] += 1

    # ------------------- Aggregated metrics across queries -------------------
    if not per_query_agg:
        print("\n[WARN] No queries evaluated in synthetic eval.")
        final_ex = final_f1 = final_ves = 0.0
    else:
        final_ex = sum(agg["ex_min"] for agg in per_query_agg.values()) / len(per_query_agg)
        final_f1 = sum(
            (agg["f1_sum"] / agg["count"]) for agg in per_query_agg.values()
        ) / len(per_query_agg)
        final_ves = sum(
            (agg["ves_sum"] / agg["count"]) for agg in per_query_agg.values()
        ) / len(per_query_agg)

    print("\n" + "=" * 80)
    print("FINAL STRICT PER-QUERY METRICS")
    print("=" * 80)
    print(f"EX  avg = {final_ex:.3f}")
    print(f"F1  avg = {final_f1:.3f}")
    print(f"VES avg = {final_ves:.3f}")

    # ------------------- JSON output (per-query, with tests) -----------------
    if out_json is None:
        stem = pred_file.stem
        out_json = pred_file.with_name(f"synthetic_eval_{stem}.json")

    # Coverage info if BIRD metadata is available
    total_bird_queries = len(bird_meta) if bird_meta else None
    num_with_tests = len(per_query_tests)
    missing_count = None
    if total_bird_queries is not None:
        missing_count = total_bird_queries - num_with_tests
        print(
            f"\n[INFO] Synthetic coverage: tests for {num_with_tests} / {total_bird_queries} "
            f"queries (missing {missing_count})."
        )

    queries_json = []
    for qid, tests in sorted(per_query_tests.items()):
        agg = per_query_agg[qid]
        meta = bird_meta.get(qid, {})

        any_gold_output = any(
            (isinstance(t["gold_output"], list) and len(t["gold_output"]) > 0)
            for t in tests
        )

        per_query_ex = agg["ex_min"]
        per_query_f1 = agg["f1_sum"] / agg["count"] if agg["count"] > 0 else 0.0
        per_query_ves = agg["ves_sum"] / agg["count"] if agg["count"] > 0 else 0.0

        queries_json.append(
            {
                "query_index": qid,
                "db_id": meta.get("db_id"),
                "difficulty": meta.get("difficulty"),
                "gold_sql": meta.get("gold_sql") or tests[0]["gold_output"],  # fallback
                "pred_sql": None if not tests else tests[0]["pred_output"],
                "any_gold_output": any_gold_output,
                "ex_all_tests": per_query_ex,
                "f1_avg": per_query_f1,
                "ves_avg": per_query_ves,
                "tests": tests,
            }
        )

    out_obj = {
        "meta": {
            "tests_dir": str(tests_dir),
            "pred_file": str(pred_file),
            "id_csv": str(id_csv) if id_csv else None,
            "bird_gold": str(bird_gold) if bird_gold else None,
            "num_queries_with_tests": num_with_tests,
            "total_bird_queries": total_bird_queries,
            "missing_queries_without_tests": missing_count,
            "final_ex_avg": final_ex,
            "final_f1_avg": final_f1,
            "final_ves_avg": final_ves,
        },
        "queries": queries_json,
    }

    with open(out_json, "w", encoding="utf8") as f:
        json.dump(out_obj, f, indent=2, ensure_ascii=False)

    print(f"\n[INFO] Synthetic eval JSON written to: {out_json}")


# -------------------------------------------------------------------
# Entrypoint
# -------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tests", required=True,
                        help="Directory containing JSON unit tests.")
    parser.add_argument("--pred", required=True,
                        help="JSON predictions file (keys are query_index).")
    parser.add_argument("--id_csv", default=None,
                        help="Optional CSV with column query_index to filter queries.")
    parser.add_argument("--bird_gold", default=None,
                        help="Optional BIRD dev.json, to attach difficulty/db_id.")
    parser.add_argument("--out_json", default=None,
                        help="Optional output JSON path for detailed per-query log.")

    args = parser.parse_args()

    run_synthetic_eval(
        args.tests,
        args.pred,
        id_csv=args.id_csv,
        bird_gold=args.bird_gold,
        out_json=args.out_json,
    )
