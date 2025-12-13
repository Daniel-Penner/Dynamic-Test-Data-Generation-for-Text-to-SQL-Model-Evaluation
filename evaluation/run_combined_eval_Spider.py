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

#EVALUATION FILE FOR COMPARISON OF SPIDER-DEV AND UNIT TEST SETS
def build_synthetic_sqlite(tables_dict):
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()

    for table_name, rows in tables_dict.items():
        if table_name.lower().startswith("sqlite_"):
            continue

        if not rows:
            continue

        cols = []
        seen = set()
        for c in rows[0].keys():
            cl = c.lower()
            if cl not in seen:
                cols.append(c)
                seen.add(cl)

        col_defs = ", ".join(f'"{c}" TEXT' for c in cols)
        cur.execute(f'CREATE TABLE "{table_name}" ({col_defs});')

        for row in rows:
            placeholders = ", ".join("?" for _ in cols)
            values = [row.get(c) for c in cols]
            cur.execute(
                f'INSERT INTO "{table_name}" VALUES ({placeholders})',
                values
            )

    conn.commit()
    return conn


def load_raw_predictions(pred_sql_file):
    """
    One SQL query per line.
    Line index = query index.
    """
    preds = {}
    with open(pred_sql_file, "r", encoding="utf8") as f:
        for i, line in enumerate(f):
            sql = line.strip()
            if sql:
                preds[i] = sql
    return preds


def load_spider_gold(gold_path):
    """
    dev_gold.sql format:
    SQL <TAB> db_id
    SQL may contain tabs, so we split on the last tab only.
    """
    out = {}
    with open(gold_path, "r", encoding="utf8") as f:
        for i, line in enumerate(f):
            sql, db_id = line.rstrip("\n").rsplit("\t", 1)
            out[i] = {
                "db_id": db_id.strip(),
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


#Core evaluation used to evaluate both queries on both sets of data
def run_spider_combined_eval(pred_path, spider_gold_path, tests_dir, out_json):

    runtime_totals = {
        "spider_ms": 0.0,
        "synthetic_ms": 0.0,
        "combined_ms": 0.0
    }

    preds = load_raw_predictions(pred_path)
    spider_gold = load_spider_gold(spider_gold_path)
    synthetic_tests = load_synthetic_tests(tests_dir)

    all_qids = sorted(set(preds) & set(spider_gold) & set(synthetic_tests))
    print(f"[INFO] Evaluating {len(all_qids)} Spider queries")

    start_time = time.perf_counter()
    per_query = []

    agg = {
        "spider_ex": 0,
        "synthetic_ex": 0,
        "combined_ex": 0,
        "f1_combined": 0.0,
        "ves_combined": 0.0,
    }

    for qid in all_qids:
        gold = spider_gold[qid]
        pred_sql = preds[qid]
        tests = synthetic_tests[qid]

        record = {
            "query_index": qid,
            "db_id": gold["db_id"],
            "spider": {},
            "synthetic": {},
            "combined": {}
        }

        #Spider
        db_path = (
            Path("spider_input_data/database")
            / gold["db_id"]
            / f"{gold['db_id']}.sqlite"
        )

        spider_start = time.perf_counter()
        gold_rows = execute_sqlite_query(str(db_path), gold["sql"])
        pred_rows = execute_sqlite_query(str(db_path), pred_sql)
        spider_runtime = (time.perf_counter() - spider_start) * 1000

        spider_ex, spider_f1, spider_ves = compare_results(pred_rows, gold_rows)

        record["spider"] = {
            "ex": spider_ex,
            "f1": spider_f1,
            "ves": spider_ves,
            "runtime_ms": spider_runtime
        }

        #Unit tests
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

        #Combined of both methods
        combined_ex = int(spider_ex == 1 and ex_all == 1)
        combined_f1 = (spider_f1 + record["synthetic"]["f1_avg"]) / 2
        combined_ves = (spider_ves + record["synthetic"]["ves_avg"]) / 2

        record["runtime"] = {
            "spider_ms": spider_runtime,
            "synthetic_ms": synth_runtime,
            "combined_ms": spider_runtime + synth_runtime
        }

        runtime_totals["spider_ms"] += spider_runtime
        runtime_totals["synthetic_ms"] += synth_runtime
        runtime_totals["combined_ms"] += spider_runtime + synth_runtime

        record["combined"] = {
            "ex": combined_ex,
            "f1": combined_f1,
            "ves": combined_ves
        }

        per_query.append(record)

        agg["spider_ex"] += spider_ex
        agg["synthetic_ex"] += ex_all
        agg["combined_ex"] += combined_ex
        agg["f1_combined"] += combined_f1
        agg["ves_combined"] += combined_ves

    wall_clock_ms = (time.perf_counter() - start_time) * 1000
    n = len(per_query)

    runtime_summary = {
        "spider_total_ms": runtime_totals["spider_ms"],
        "synthetic_total_ms": runtime_totals["synthetic_ms"],
        "combined_total_ms": runtime_totals["combined_ms"],

        "spider_avg_ms": runtime_totals["spider_ms"] / n if n else 0.0,
        "synthetic_avg_ms": runtime_totals["synthetic_ms"] / n if n else 0.0,
        "combined_avg_ms": runtime_totals["combined_ms"] / n if n else 0.0,

        "wall_clock_ms": wall_clock_ms
    }

    summary = {
        "num_queries": n,
        "spider_ex_avg": agg["spider_ex"] / n if n else 0.0,
        "synthetic_ex_avg": agg["synthetic_ex"] / n if n else 0.0,
        "combined_ex_avg": agg["combined_ex"] / n if n else 0.0,
        "f1_combined_avg": agg["f1_combined"] / n if n else 0.0,
        "ves_combined_avg": agg["ves_combined"] / n if n else 0.0,
        "runtime": runtime_summary
    }

    out = {
        "meta": {
            "pred_path": pred_path,
            "spider_gold_path": spider_gold_path,
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
    print("SPIDER COMBINED EXECUTION EVALUATION SUMMARY")
    print("=" * 80)
    for k, v in summary.items():
        if k != "runtime":
            print(f"{k:25s}: {v:.4f}")
    print("\nRUNTIME SUMMARY (ms)")
    for k, v in runtime_summary.items():
        print(f"{k:25s}: {v:.2f}")

    print(f"\n[INFO] Output written to: {out_json}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred", required=True, help="Raw SQL file (1 query per line)")
    parser.add_argument("--spider_gold", required=True)
    parser.add_argument("--tests", required=True)
    parser.add_argument("--out_json", required=True)

    args = parser.parse_args()

    run_spider_combined_eval(
        pred_path=args.pred,
        spider_gold_path=args.spider_gold,
        tests_dir=args.tests,
        out_json=args.out_json
    )
