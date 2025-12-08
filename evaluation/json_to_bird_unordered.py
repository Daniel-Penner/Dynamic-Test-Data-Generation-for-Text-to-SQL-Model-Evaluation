import json
from pathlib import Path

# -----------------------------
# Paths
# -----------------------------
PRED_JSON = Path("prediction_queries/Alpha-SQL+Qwen_32b.json")  # input
BIRD_JSON = Path("bird_input_data/dev.json")
OUT_FILE = Path("prediction_queries/Alpha-SQL+Qwen_32b_predictions.json")

SEPARATOR = "\t----- bird -----\t"


def normalize_sql(sql: str) -> str:
    sql = sql.strip()
    if not sql.endswith(";"):
        sql += ";"
    return sql


def main():
    # Load model predictions (dict: qid -> SQL)
    with open(PRED_JSON, "r", encoding="utf-8") as f:
        preds = json.load(f)

    if not isinstance(preds, dict):
        raise ValueError("Expected prediction file to be a JSON dict keyed by query id.")

    # Load BIRD gold metadata
    with open(BIRD_JSON, "r", encoding="utf-8") as f:
        bird_data = json.load(f)

    # Build lookup: index -> db_id
    bird_db_lookup = {
        str(i): entry["db_id"]
        for i, entry in enumerate(bird_data)
    }

    out = {}

    for qid, sql in preds.items():
        if qid not in bird_db_lookup:
            raise KeyError(f"Query id {qid} not found in BIRD dev.json")

        sql = normalize_sql(sql)
        db_id = bird_db_lookup[qid]

        out[qid] = f"{sql}{SEPARATOR}{db_id}"

    # Ensure output directory exists
    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    with open(OUT_FILE, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=4, ensure_ascii=False)

    print(f"[OK] Wrote {len(out)} labeled queries → {OUT_FILE}")


if __name__ == "__main__":
    main()
