import csv
from pathlib import Path

#ONE OF MANY CONVERSION FILES FOR PREDICTION SQL WHICH DIFFERENT MODELS HAVE PRESENTED IN DIFFERENT FORMS
INPUT_CSV = Path("prediction_queries/Spider/DIN-SQL+CodeX.csv") #Input manually set to one of the evaluated model prediction files
OUTPUT_SQL = Path("prediction_queries/Spider/DIN-SQL+CodeX_predictions.sql")

def normalize_sql(sql: str) -> str:
    sql = sql.strip()
    return sql if sql.endswith(";") else sql + ";"


def convert_csv_to_queries(input_csv: Path, output_file: Path):
    with input_csv.open(newline="", encoding="utf-8") as csvfile, \
         output_file.open("w", encoding="utf-8") as out:

        reader = csv.DictReader(csvfile)

        for row in reader:
            predicted = row["PREDICTED SQL"].strip()
            if not predicted:
                continue

            out.write(normalize_sql(predicted) + "\n")

    print(f"[✓] Wrote {output_file}")


if __name__ == "__main__":
    convert_csv_to_queries(INPUT_CSV, OUTPUT_SQL)
