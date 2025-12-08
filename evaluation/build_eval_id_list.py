# build_eval_id_list.py
import argparse
import json
import csv
from pathlib import Path


def main(tests_dir, bird_gold_path, out_csv):
    tests_dir = Path(tests_dir)

    # 1) Collect all query_index values that have at least one test
    qids_with_tests = set()
    for tf in tests_dir.rglob("*.json"):
        with open(tf, "r", encoding="utf8") as f:
            obj = json.load(f)
        qids_with_tests.add(int(obj["query_index"]))

    # 2) Load BIRD dev.json metadata
    with open(bird_gold_path, "r", encoding="utf8") as f:
        data = json.load(f)

    total_queries = len(data)

    # 3) Write CSV
    out_csv = Path(out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    with open(out_csv, "w", encoding="utf8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["query_index", "db_id", "difficulty", "gold_sql"],
        )
        writer.writeheader()

        for qid in sorted(qids_with_tests):
            if qid >= total_queries:
                print(f"[WARN] query_index {qid} >= len(bird_gold); skipping.")
                continue

            q = data[qid]
            sql = (
                q.get("query")
                or q.get("SQL")
                or q.get("gold")
                or q.get("gold_query")
            )

            writer.writerow(
                {
                    "query_index": qid,
                    "db_id": q["db_id"],
                    "difficulty": q.get("difficulty"),
                    "gold_sql": sql.strip() if sql else "",
                }
            )

    # 4) Coverage info
    num_with_tests = len(
        [qid for qid in qids_with_tests if qid < total_queries]
    )
    missing = total_queries - num_with_tests

    print(f"[INFO] Total BIRD queries:    {total_queries}")
    print(f"[INFO] Queries with tests:    {num_with_tests}")
    print(f"[INFO] Queries with no tests: {missing}")
    print(f"[INFO] CSV written to:        {out_csv}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tests", required=True,
                        help="Directory of synthetic test JSON files.")
    parser.add_argument("--bird_gold", required=True,
                        help="Path to BIRD dev.json.")
    parser.add_argument("--out_csv", required=True,
                        help="Output CSV path (will contain query_index, db_id, difficulty, gold_sql).")

    args = parser.parse_args()
    main(args.tests, args.bird_gold, args.out_csv)
