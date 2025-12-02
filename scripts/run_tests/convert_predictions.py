import json
import re
from pathlib import Path
import sys


def clean_prediction_line(line: str):
    """
    Extracts ID, SQL, and DB name from lines such as:
    "0": "SELECT ... ;\t----- bird -----\tcalifornia_schools",
    """

    # Extract ID
    id_match = re.match(r'\s*"(\d+)"\s*:', line)
    if not id_match:
        return None
    pred_id = int(id_match.group(1))

    # Extract content inside: " ... "
    content_match = re.search(r'":\s*"(.*)"', line)
    if not content_match:
        return None
    content = content_match.group(1)

    # Split on the pattern used in DeepSeek output: ----- bird ----- db_id
    parts = content.split("\t----- bird -----\t")

    if len(parts) != 2:
        # If db_id not found, skip this line
        return None

    sql = parts[0].strip().rstrip(";")  # remove trailing semicolon
    db_id = parts[1].strip()

    return {
        "id": pred_id,
        "db_id": db_id,
        "sql": sql
    }


def convert_file(input_path: str):
    input_path = Path(input_path)
    output_path = input_path.parent / f"cleaned_{input_path.stem}.json"

    with open(input_path, "r", encoding="utf-8") as f:
        raw = f.read()

    # The file itself may be JSON-like; try to load it first.
    try:
        data = json.loads(raw)
        # It's already a dict { "0": " ... " , ... }
        lines = []
        for k, v in data.items():
            combined = f'"{k}": "{v}"'
            lines.append(combined)
    except:
        # Otherwise, treat each raw line independently
        lines = raw.splitlines()

    clean_items = []
    for line in lines:
        cleaned = clean_prediction_line(line)
        if cleaned:
            clean_items.append(cleaned)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(clean_items, f, indent=2)

    print(f"✔ Cleaned file written to:\n{output_path}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python convert_predictions.py <input_file>")
        sys.exit(1)

    convert_file(sys.argv[1])
