# scripts/run_tests/run_bird_eval.py

import json
import argparse
from pathlib import Path
import duckdb
import pandas as pd


def load_predictions(path):
    preds = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            preds[obj["query_id"]] = obj["sql"]
    return preds


def execute_sql(conn, sql):
    try:
        df = conn.execute(sql).fetchdf()
        return df, None
    except Exception as e:
        return None, str(e)


def exact_match(df_pred, df_gold):
    if df_pred is None:
        return False
    return df_pred.equals(df_gold)


def soft_f1(df_pred, df_gold):
    if df_pred is None or df_gold is None:
        return 0.0

    pred = set(map(tuple, df_pred.to_numpy()))
    gold = set(map(tuple, df_gold.to_numpy()))

    tp = len(pred & gold)
    fp = len(pred - gold)
    fn = len(gold - pred)

    if tp == 0:
        return 0.0

    precision = tp / (tp + fp) if (tp + fp) else 0
    recall    = tp / (tp + fn) if (tp + fn) else 0

    if precision + recall == 0:
        return 0.0

    return 2 * precision * recall / (precision + recall)


def ves(df_pred, df_gold):
    """Row-level cosine similarity on value vectors (BIRD metric)."""
    if df_pred is None or df_gold is None:
        return 0.0

    if len(df_pred) == 0 or len(df_gold) == 0:
        return 0.0

    import numpy as np
    from numpy.linalg import norm

    p = df_pred.to_numpy().flatten().astype(str)
    g = df_gold.to_numpy().flatten().astype(str)

    # Pad to same length
    L = max(len(p), len(g))
    p = np.pad(p, (0, L - len(p)), constant_values="")
    g = np.pad(g, (0, L - len(g)), constant_values="")

    # Bag-of-values cosine similarity
    def vec(values):
        return pd.Series(values).value_counts().sort_index()

    p_vec = vec(p)
    g_vec = vec(g)

    # Align index
    p_vec, g_vec = p_vec.align(g_vec, fill_value=0)

    num = (p_vec * g_vec).sum()
    denom = norm(p_vec) * norm(g_vec)

    return float(num / denom) if denom else 0.0


def main(pred_file, bird_root):
    preds = load_predictions(pred_file)

    # Load gold dev dataset
    with open(bird_root / "dev.json", "r") as f:
        gold_queries = {q["query_id"]: q for q in json.load(f)}

    ea_scores = []
    soft_f1_scores = []
    ves_scores = []

    for qid, gold in gold_queries.items():
        if qid not in preds:
            continue

        sql_pred = preds[qid]
        sql_gold = gold["SQL"]
        db_path = bird_root / "dev-database" / gold["db_id"] / f"{gold['db_id']}.sqlite"

        conn = duckdb.connect()
        conn.execute(f"ATTACH '{db_path}' AS db")

        df_gold, err1 = execute_sql(conn, sql_gold)
        df_pred, err2 = execute_sql(conn, sql_pred)

        ea = 1 if exact_match(df_pred, df_gold) else 0
        sf = soft_f1(df_pred, df_gold)
        vs = ves(df_pred, df_gold)

        ea_scores.append(ea)
        soft_f1_scores.append(sf)
        ves_scores.append(vs)

        print(f"Query {qid}: EA={ea} Soft-F1={sf:.3f} VES={vs:.3f}")

    print("\n==============================")
    print("BIRD Evaluation Summary")
    print("==============================")
    print(f"Execution Accuracy: {sum(ea_scores)/len(ea_scores):.3f}")
    print(f"Mean Soft-F1:       {sum(soft_f1_scores)/len(soft_f1_scores):.3f}")
    print(f"Mean VES:           {sum(ves_scores)/len(ves_scores):.3f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred", required=True, help="Path to model prediction file (JSONL).")
    parser.add_argument("--bird", required=True, help="Path to bird_input_data directory.")
    args = parser.parse_args()

    main(Path(args.pred), Path(args.bird))
