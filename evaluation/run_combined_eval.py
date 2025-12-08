import argparse
import json
import time
import sqlite3
from pathlib import Path
from collections import defaultdict

from evaluation.evaluator_core import (
    execute_sqlite_query,
    execute_sqlite_query_conn,
    compare_results
)

# ---------------------------------------------------------
# Synthetic DB builder (copied explicitly για clarity)
# ---------------------------------------------------------
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
                values
            )

    conn.commit()
    return conn


# ---------------------------------------------------------
# Loaders
# ---------------------------------------------------------
def load_predictions(pred_path):
    with open(pred_path, "r", encoding="utf8") as f:
        raw = json.load(f)

    preds = {}
    for k, v in raw.items():
        sql = v.split("\t-----")[0].strip()
        preds[int(k)] = sql

    return preds


def load_bird_gold(bird_gold_path):
    with open(bird_gold_path, "r", encoding="utf8") as f:
        data = json.load(f)

    out = {}
    for i, q in enumerate(data):
        sql = (
            q.get("query")
            or q.get("SQL")
            or q.get("gold")
            or q.get("gold_query")
        )
        out[i] = {
            "db_id": q["db_id"],
            "difficulty": q.get("difficulty"),
            "sql": sql.strip()
        }
    return out


def load_synthetic_tests(tests_dir):
    tests = defaultdict(list)
    for p in Path(tests_dir).rglob("*.json"):
        with open(p, "r", encoding="utf8") as f:
            obj = json.load(f)
        qid = int(obj["query_index"])
        tests[qid].append(obj)
    return tests


# ---------------------------------------------------------
# Core combined evaluation
# ---------------------------------------------------------
def run_combined_eval(pred_path, bird_gold_path, tests_dir, out_json):

    runtime_totals = {
        "bird_ms": 0.0,
        "synthetic_ms": 0.0,
        "combined_ms": 0.0
    }

    preds = load_predictions(pred_path)
    bird_gold = load_bird_gold(bird_gold_path)
    synthetic_tests = load_synthetic_tests(tests_dir)

    all_qids = sorted(set(preds) & set(bird_gold) & set(synthetic_tests))

    print(f"[INFO] Evaluating {len(all_qids)} queries")

    start_time = time.perf_counter()

    per_query = []

    agg = {
        "bird_ex": 0,
        "synthetic_ex": 0,
        "combined_ex": 0,
        "f1_combined": 0.0,
        "ves_combined": 0.0,
    }

    for qid in all_qids:
        gold = bird_gold[qid]
        pred_sql = preds[qid]
        tests = synthetic_tests[qid]

        record = {
            "query_index": qid,
            "db_id": gold["db_id"],
            "difficulty": gold["difficulty"],
            "bird": {},
            "synthetic": {},
            "combined": {}
        }

        # ---------------- BIRD ----------------
        db_path = (
            Path("bird_input_data/dev_databases")
            / gold["db_id"]
            / f"{gold['db_id']}.sqlite"
        )

        bird_start = time.perf_counter()
        gold_rows = execute_sqlite_query(str(db_path), gold["sql"])
        pred_rows = execute_sqlite_query(str(db_path), pred_sql)
        bird_runtime = (time.perf_counter() - bird_start) * 1000

        bird_ex, bird_f1, bird_ves = compare_results(pred_rows, gold_rows)

        record["bird"] = {
            "ex": bird_ex,
            "f1": bird_f1,
            "ves": bird_ves,
            "runtime_ms": bird_runtime
        }

        # ---------------- SYNTHETIC ----------------
        synth_start = time.perf_counter()

        ex_all = 1
        f1_sum = 0.0
        ves_sum = 0.0
        test_outputs = []

        for t in tests:
            conn = build_synthetic_sqlite(t["tables"])
            gold_out = execute_sqlite_query_conn(conn, t["sql"])
            pred_out = execute_sqlite_query_conn(conn, pred_sql)

            ex, f1, ves = compare_results(pred_out, gold_out)

            test_outputs.append({
                "test_name": t.get("name"),
                "gold_error": isinstance(gold_out, dict),
                "pred_error": isinstance(pred_out, dict),
                "ex": ex,
                "f1": f1,
                "ves": ves,
                "gold_output": gold_out,
                "pred_output": pred_out
            })

            ex_all &= ex
            f1_sum += f1
            ves_sum += ves

            conn.close()

        synth_runtime = (time.perf_counter() - synth_start) * 1000
        k = len(tests)

        record["synthetic"] = {
            "ex_all_tests": ex_all,
            "f1_avg": f1_sum / k if k else 0.0,
            "ves_avg": ves_sum / k if k else 0.0,
            "runtime_ms": synth_runtime,
            "tests": test_outputs
        }

        # ---------------- COMBINED ----------------
        combined_ex = int(bird_ex == 1 and ex_all == 1)
        combined_f1 = (bird_f1 + record["synthetic"]["f1_avg"]) / 2
        combined_ves = (bird_ves + record["synthetic"]["ves_avg"]) / 2

        record["runtime"] = {
            "bird_ms": bird_runtime,
            "synthetic_ms": synth_runtime,
            "combined_ms": bird_runtime + synth_runtime
        }
        runtime_totals["bird_ms"] += bird_runtime
        runtime_totals["synthetic_ms"] += synth_runtime
        runtime_totals["combined_ms"] += bird_runtime + synth_runtime

        record["combined"] = {
            "ex": combined_ex,
            "f1": combined_f1,
            "ves": combined_ves
        }

        per_query.append(record)

        agg["bird_ex"] += bird_ex
        agg["synthetic_ex"] += ex_all
        agg["combined_ex"] += combined_ex
        agg["f1_combined"] += combined_f1
        agg["ves_combined"] += combined_ves


    wall_clock_ms = (time.perf_counter() - start_time) * 1000
    n = len(per_query)

    runtime_summary = {
        "bird_total_ms": runtime_totals["bird_ms"],
        "synthetic_total_ms": runtime_totals["synthetic_ms"],
        "combined_total_ms": runtime_totals["combined_ms"],

        "bird_avg_ms": runtime_totals["bird_ms"] / n if n else 0.0,
        "synthetic_avg_ms": runtime_totals["synthetic_ms"] / n if n else 0.0,
        "combined_avg_ms": runtime_totals["combined_ms"] / n if n else 0.0,

        # optional, but useful to keep
        "wall_clock_ms": wall_clock_ms
    }

    summary = {
        "num_queries": n,

        "bird_ex_avg": agg["bird_ex"] / n if n else 0.0,
        "synthetic_ex_avg": agg["synthetic_ex"] / n if n else 0.0,
        "combined_ex_avg": agg["combined_ex"] / n if n else 0.0,

        "f1_combined_avg": agg["f1_combined"] / n if n else 0.0,
        "ves_combined_avg": agg["ves_combined"] / n if n else 0.0,

        "runtime": runtime_summary
    }


    out = {
        "meta": {
            "pred_path": pred_path,
            "bird_gold_path": bird_gold_path,
            "tests_dir": tests_dir,
            **summary
        },
        "queries": per_query
    }

    out_path = Path(out_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, "w", encoding="utf8") as f:
        json.dump(out, f, indent=2)

    print("\n" + "=" * 80)
    print("COMBINED EXECUTION EVALUATION SUMMARY")
    print("=" * 80)
    print(f"{'num_queries':22s}: {summary['num_queries']}")
    print(f"{'bird_ex_avg':22s}: {summary['bird_ex_avg']:.4f}")
    print(f"{'synthetic_ex_avg':22s}: {summary['synthetic_ex_avg']:.4f}")
    print(f"{'combined_ex_avg':22s}: {summary['combined_ex_avg']:.4f}")
    print(f"{'f1_combined_avg':22s}: {summary['f1_combined_avg']:.4f}")
    print(f"{'ves_combined_avg':22s}: {summary['ves_combined_avg']:.4f}")

    print("\nRUNTIME SUMMARY (ms)")
    for k, v in summary["runtime"].items():
        print(f"{k:22s}: {v:.2f}")

    print(f"\n[INFO] Output written to: {out_json}")


# ---------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred", required=True)
    parser.add_argument("--bird_gold", required=True)
    parser.add_argument("--tests", required=True)
    parser.add_argument("--out_json", required=True)

    args = parser.parse_args()

    run_combined_eval(
        pred_path=args.pred,
        bird_gold_path=args.bird_gold,
        tests_dir=args.tests,
        out_json=args.out_json
    )
