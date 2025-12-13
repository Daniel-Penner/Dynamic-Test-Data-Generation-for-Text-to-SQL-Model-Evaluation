import json
from pathlib import Path

#ONE OF MANY CONVERSION FILES FOR PREDICTION SQL WHICH DIFFERENT MODELS HAVE PRESENTED IN DIFFERENT FORMS
PRED_JSON = Path("prediction_queries/BIRD/OmniSQL_7b_greedy.json") #Input manually set to one of the evaluated model prediction files
BIRD_JSON = Path("bird_input_data/dev.json")
OUT_FILE = Path("prediction_queries/BIRD/OmniSQL_7b_greedy_predictions.json")

SEPARATOR = "\t----- bird -----\t"


def normalize_sql(sql: str) -> str:
    sql = sql.strip()
    if not sql.endswith(";"):
        sql += ";"
    return sql


def main():
    with open(PRED_JSON, "r", encoding="utf-8") as f:
        sql_list = json.load(f)

    if not isinstance(sql_list, list):
        raise ValueError("Expected OmniSQL prediction file to be a JSON list.")

    with open(BIRD_JSON, "r", encoding="utf-8") as f:
        bird_data = json.load(f)

    if len(sql_list) != len(bird_data):
        raise ValueError(
            f"Prediction count ({len(sql_list)}) != BIRD entries ({len(bird_data)})"
        )

    out = {}

    for i, (sql, bird_entry) in enumerate(zip(sql_list, bird_data)):
        qid = str(i)
        db_id = bird_entry["db_id"]

        sql = normalize_sql(sql)
        out[qid] = f"{sql}{SEPARATOR}{db_id}"

    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    with open(OUT_FILE, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=4, ensure_ascii=False)

    print(f"[OK] Wrote {len(out)} queries → {OUT_FILE}")


if __name__ == "__main__":
    main()
