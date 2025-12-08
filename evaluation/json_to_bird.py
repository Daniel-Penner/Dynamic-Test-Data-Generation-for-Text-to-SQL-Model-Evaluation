import json
from pathlib import Path

# -----------------------------
# Paths
# -----------------------------
PRED_JSON = Path("prediction_queries/OmniSQL_7b_greedy.json")
BIRD_JSON = Path("bird_input_data/dev.json")
OUT_FILE = Path("prediction_queries/OmniSQL_7b_greedy_predictions.json")

SEPARATOR = "\t----- bird -----\t"


def normalize_sql(sql: str) -> str:
    sql = sql.strip()
    if not sql.endswith(";"):
        sql += ";"
    return sql


def main():
    # Load OmniSQL predictions (list of SQL strings)
    with open(PRED_JSON, "r", encoding="utf-8") as f:
        sql_list = json.load(f)

    if not isinstance(sql_list, list):
        raise ValueError("Expected OmniSQL prediction file to be a JSON list.")

    # Load BIRD gold metadata
    with open(BIRD_JSON, "r", encoding="utf-8") as f:
        bird_data = json.load(f)

    if len(sql_list) != len(bird_data):
        raise ValueError(
            f"Prediction count ({len(sql_list)}) != BIRD entries ({len(bird_data)})"
        )

    out = {}

    # IMPORTANT: enumerate by index to match evaluation scripts
    for i, (sql, bird_entry) in enumerate(zip(sql_list, bird_data)):
        qid = str(i)  # ✔ evaluator expects query_index-based alignment
        db_id = bird_entry["db_id"]

        sql = normalize_sql(sql)
        out[qid] = f"{sql}{SEPARATOR}{db_id}"

    # Ensure output directory exists
    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    with open(OUT_FILE, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=4, ensure_ascii=False)

    print(f"[OK] Wrote {len(out)} queries → {OUT_FILE}")


if __name__ == "__main__":
    main()
