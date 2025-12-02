import json
import duckdb
import argparse
from pathlib import Path
from collections import defaultdict
import pandas as pd


# -------------------------------------------------------------
# Helpers
# -------------------------------------------------------------
def normalize_sql(sql: str) -> str:
    """Convert MySQL-style backticks → DuckDB-compatible double quotes."""
    if sql is None:
        return sql
    return sql.replace("`", '"')


def run_sql(conn, sql):
    """Execute SQL and capture errors safely."""
    try:
        df = conn.execute(sql).fetchdf()
        return df.to_dict(orient="records"), None
    except Exception as e:
        return None, str(e)


def results_match(pred, gold):
    """Exact output match for synthetic data."""
    if pred is None:
        return False
    return pred == gold


def load_predictions(pred_file):
    """Load model predictions indexed by query id."""
    with open(pred_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    return {item["id"]: item for item in data}


# -------------------------------------------------------------
# Evaluate a single query across its scenarios
# -------------------------------------------------------------
def evaluate_query(query_id, scenario_files, prediction_sql):
    passed = 0
    total = len(scenario_files)
    scenario_details = []

    prediction_sql = normalize_sql(prediction_sql)

    for file in scenario_files:
        with open(file, "r", encoding="utf-8") as f:
            scenario = json.load(f)

        gold_sql = normalize_sql(scenario["sql"])
        tables = scenario["tables"]
        expected_output = scenario.get("expected_output", None)

        # Build database
        conn = duckdb.connect()
        for tbl_name, rows in tables.items():
            df = pd.DataFrame(rows)
            conn.register(tbl_name, df)

        # Run gold SQL
        gold_result, gold_err = run_sql(conn, gold_sql)
        if gold_err:
            print(f"[ERROR] Gold SQL failed for scenario '{scenario.get('scenario')}'")
            print(gold_err)
            gold_result = None

        # Use expected_output as ground truth when present
        if expected_output is not None:
            gold_result = expected_output

        # Run predicted SQL
        pred_result, pred_err = run_sql(conn, prediction_sql)

        ok = results_match(pred_result, gold_result)
        if ok:
            passed += 1

        scenario_details.append({
            "scenario": scenario.get("scenario"),
            "gold_sql": gold_sql,
            "prediction_sql": prediction_sql,
            "passed": ok,
            "predicted": pred_result,
            "gold": gold_result,
            "error_pred": pred_err,
            "error_gold": gold_err,
        })

    strict_acc = 1 if passed == total else 0
    frac_acc = passed / total if total > 0 else 0.0
    return strict_acc, frac_acc, scenario_details


# -------------------------------------------------------------
# Main: run all tests
# -------------------------------------------------------------
def main(test_dir, pred_file):
    test_dir = Path(test_dir)
    predictions = load_predictions(pred_file)

    # Load all JSON files
    test_files = list(test_dir.glob("*.json"))
    if not test_files:
        print("No test files found. Check directory path.")
        return

    # Group by query_index
    grouped = defaultdict(list)
    for f in test_files:
        with open(f, "r", encoding="utf-8") as jf:
            data = json.load(jf)
        qid = data["query_index"]
        grouped[qid].append(f)

    all_results = []
    strict_scores = []
    frac_scores = []

    print("====================================================")
    print("Synthetic Test Results")
    print("====================================================")

    # Evaluate each query
    for qid, scenario_files in sorted(grouped.items()):
        if qid not in predictions:
            print(f"WARNING: No prediction for query {qid}. Skipping.")
            continue

        prediction_sql = predictions[qid]["sql"]

        strict_acc, frac_acc, details = evaluate_query(qid, scenario_files, prediction_sql)

        strict_scores.append(strict_acc)
        frac_scores.append(frac_acc)

        all_results.append({
            "query_id": qid,
            "strict_accuracy": strict_acc,
            "fractional_accuracy": frac_acc,
            "scenarios": details
        })

        print(f"\nQuery {qid}: strict={strict_acc} frac={frac_acc:.2f}")
        for d in details:
            print(f"  - Scenario {d['scenario']}: {'PASS' if d['passed'] else 'FAIL'}")

    # Summary
    print("\n====================================================")
    print(f"Queries evaluated: {len(all_results)}")
    if strict_scores:
        print(f"Execution Accuracy (strict): {sum(strict_scores)/len(strict_scores):.3f}")
        print(f"Mean Fractional Accuracy:    {sum(frac_scores)/len(frac_scores):.3f}")
    print("====================================================")

    # Write results
    output_path =  "test_results/synthetic_eval_results.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)

    print(f"\nSaved detailed results → {output_path}")


# -------------------------------------------------------------
# CLI
# -------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tests", required=True, help="Directory of synthetic test case JSON files.")
    parser.add_argument("--predictions", required=True, help="JSON file with model SQL predictions.")
    args = parser.parse_args()
    main(args.tests, args.predictions)
