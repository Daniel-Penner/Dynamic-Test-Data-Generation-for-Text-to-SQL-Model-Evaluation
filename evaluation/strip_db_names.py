from pathlib import Path

INPUT_FILE = Path("prediction_queries\Spider\Graphix-3B+PICARD.sql")
OUTPUT_FILE = Path("prediction_queries\Spider\Graphix-3B+PICARD_predictions.sql")

def strip_database_suffix(line: str) -> str:
    line = line.rstrip()

    # Split on whitespace; last token is DB name
    parts = line.split()
    if len(parts) <= 1:
        return line

    return " ".join(parts[:-1])


def process_file(input_path: Path, output_path: Path):
    with input_path.open("r", encoding="utf-8") as fin, \
         output_path.open("w", encoding="utf-8") as fout:

        for line in fin:
            line = line.strip()
            if not line:
                continue

            fout.write(strip_database_suffix(line) + "\n")

    print(f"[✓] Database names stripped → {output_path}")


if __name__ == "__main__":
    process_file(INPUT_FILE, OUTPUT_FILE)
