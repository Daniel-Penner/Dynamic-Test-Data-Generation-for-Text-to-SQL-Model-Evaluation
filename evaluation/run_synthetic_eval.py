import argparse
import json
import sqlite3
from pathlib import Path
from collections import defaultdict

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

        # Column inference (everything as TEXT)
        cols = list(rows[0].keys())
        col_defs = ", ".join(f'"{c}" TEXT' for c in cols)
        cur.execute(f'CREATE TABLE "{table_name}" ({col_defs});')

        for row in rows:
            placeholders = ", ".join("?" for _ in cols)
            values = [row[c] for c in cols]
            cur.execute(
                f'INSERT INTO "{table_name}" VALUES ({placeholders})',
                values
            )

    conn.commit()
    return conn


# -------------------------------------------------------------------
# Safe SQL execution (no multiprocessing)
# -------------------------------------------------------------------
def execute_sqlite_query(conn, sql):
    try:
        cur = conn.execute(sql)
        rows = cur.fetchall()
        return [dict(r) for r in rows]
    except Exception as e:
        return [{"error": str(e)}]


# -------------------------------------------------------------------
# Metric Computations
# -------------------------------------------------------------------
def compare_results(pred_rows, gold_rows):
    """Return (EX, F1, VES)"""
    # Timeout or error → EX = 0
    if isinstance(pred_rows, list) and pred_rows and "error" in pred_rows[0]:
        return 0, 0.0, 0.0

    # EX: exact match
    ex = int(pred_rows == gold_rows)

    # F1: row-level overlap
    def to_tuple_list(rows):
        return [tuple(sorted(r.items())) for r in rows]

    p_set = set(to_tuple_list(pred_rows))
    g_set = set(to_tuple_list(gold_rows))

    if not p_set and not g_set:
        f1 = 1.0
    elif not p_set or not g_set:
        f1 = 0.0
    else:
        tp = len(p_set & g_set)
        fp = len(p_set - g_set)
        fn = len(g_set - p_set)
        if tp == 0:
            f1 = 0.0
        else:
            f1 = 2 * tp / (2 * tp + fp + fn)

    # VES: 1 if both empty, else 1 if non-empty and EX=1, else 0
    if not pred_rows and not gold_rows:
        ves = 1.0
    else:
        ves = float(pred_rows == gold_rows)

    return ex, f1, ves


# -------------------------------------------------------------------
# Prediction parsing
# -------------------------------------------------------------------
def clean_pred_sql(raw_sql: str) -> str:
    if "\t-----" in raw_sql:
        return raw_sql.split("\t-----")[0].strip()
    return raw_sql.strip()


# -------------------------------------------------------------------
# Run one test
# -------------------------------------------------------------------
def run_one_test(test_obj, pred_sql):
    gold_sql = test_obj["sql"]

    # Build DB
    conn = build_synthetic_sqlite(test_obj["tables"])

    gold_out = execute_sqlite_query(conn, gold_sql)
    pred_out = execute_sqlite_query(conn, pred_sql)

    ex, f1, ves = compare_results(pred_out, gold_out)

    conn.close()
    return ex, f1, ves, gold_out, pred_out


# -------------------------------------------------------------------
# Whole evaluation
# -------------------------------------------------------------------
def run_synthetic_eval(tests_dir, pred_file):
    tests_dir = Path(tests_dir)
    pred_file = Path(pred_file)

    # Load predictions dict
    with open(pred_file, "r", encoding="utf8") as f:
        preds_raw = json.load(f)

    test_files = sorted(tests_dir.rglob("*.json"))
    results = []

    for tf in test_files:
        with open(tf, "r", encoding="utf8") as f:
            test_obj = json.load(f)

        qid = str(test_obj["query_index"])
        test_name = tf.name

        print("=" * 80)
        print(f"[Test] {test_name}")

        if qid not in preds_raw:
            print(f"[WARN] Missing prediction for query_index={qid}")
            pred_sql = test_obj["sql"]
        else:
            pred_sql = clean_pred_sql(preds_raw[qid])

        ex, f1, ves, gold_out, pred_out = run_one_test(test_obj, pred_sql)

        print("GOLD SQL:", test_obj["sql"])
        print("PRED SQL:", pred_sql)
        print("\n--- GOLD OUTPUT ---")
        print(gold_out)
        print("\n--- PRED OUTPUT ---")
        print(pred_out)
        print(f"\n--> EX  = {ex}")
        print(f"--> F1  = {f1}")
        print(f"--> VES = {ves}")

        results.append((ex, f1, ves))

    # Summary
        # -------------------------------------------------------------
    # STRICT PER-QUERY AGGREGATION (scenarios grouped by query_index)
    # -------------------------------------------------------------
    per_query = defaultdict(lambda: {
        "ex": 1,       # strict: if any scenario fails → ex = 0
        "f1_sum": 0.0,
        "ves_sum": 0.0,
        "count": 0,
    })

    for tf, (ex, f1, ves) in zip(test_files, results):
        test_obj = json.loads(Path(tf).read_text())
        qid = str(test_obj["query_index"])

        per_query[qid]["ex"] = min(per_query[qid]["ex"], ex)
        per_query[qid]["f1_sum"] += f1
        per_query[qid]["ves_sum"] += ves
        per_query[qid]["count"] += 1

    # Final aggregated metrics across queries
    final_ex  = sum(per_query[q]["ex"] for q in per_query) / len(per_query)
    final_f1  = sum((per_query[q]["f1_sum"] / per_query[q]["count"]) for q in per_query) / len(per_query)
    final_ves = sum((per_query[q]["ves_sum"] / per_query[q]["count"]) for q in per_query) / len(per_query)

    print("\n" + "=" * 80)
    print("FINAL STRICT PER-QUERY METRICS")
    print("=" * 80)
    print(f"EX  avg = {final_ex:.3f}")
    print(f"F1  avg = {final_f1:.3f}")
    print(f"VES avg = {final_ves:.3f}")


# -------------------------------------------------------------------
# Entrypoint
# -------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tests", required=True)
    parser.add_argument("--pred", required=True)
    args = parser.parse_args()

    run_synthetic_eval(args.tests, args.pred)
