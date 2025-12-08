import json
from pathlib import Path

SQL_FILE = Path("prediction_queries/RSL-SQL+deepseek.sql")
BIRD_JSON = Path("bird_input_data/dev.json")
OUT_FILE = Path("prediction_queries/RSL-SQL+deepseek_predictions.json")

SEPARATOR = "\t----- bird -----\t"


def load_sql_lines(sql_path: Path):
    lines = []
    with open(sql_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if not line.endswith(";"):
                line += ";"
            lines.append(line)
    return lines


def main():
    sql_lines = load_sql_lines(SQL_FILE)

    with open(BIRD_JSON, "r", encoding="utf-8") as f:
        bird_data = json.load(f)

    if len(sql_lines) != len(bird_data):
        raise ValueError(
            f"SQL lines ({len(sql_lines)}) != BIRD entries ({len(bird_data)})"
        )

    out = {}

    for i, (sql, bird_entry) in enumerate(zip(sql_lines, bird_data)):
        qid = str(bird_entry["question_id"])
        db_id = bird_entry["db_id"]

        out[qid] = f"{sql}{SEPARATOR}{db_id}"

    with open(OUT_FILE, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=4, ensure_ascii=False)

    print(f"[OK] Wrote {len(out)} queries → {OUT_FILE}")


if __name__ == "__main__":
    main()
