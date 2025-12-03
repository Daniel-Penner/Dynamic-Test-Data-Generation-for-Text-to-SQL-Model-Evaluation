# db_builder_synthetic.py
import sqlite3

def infer_sqlite_type(value):
    if value is None:
        return "TEXT"  # can't infer
    if isinstance(value, bool):
        return "INTEGER"
    if isinstance(value, int):
        return "INTEGER"
    if isinstance(value, float):
        return "REAL"
    return "TEXT"

def build_synthetic_sqlite(test_json):
    conn = sqlite3.connect(":memory:", check_same_thread=False)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()

    for table_name, rows in test_json["tables"].items():
        if len(rows) == 0:
            continue

        # Infer columns + types
        sample_row = rows[0]
        cols = list(sample_row.keys())
        col_defs = []

        for c in cols:
            inferred = infer_sqlite_type(sample_row[c])
            col_defs.append(f'"{c}" {inferred}')

        cur.execute(f'CREATE TABLE "{table_name}" ({", ".join(col_defs)});')

        # Insert rows
        for row in rows:
            placeholders = ", ".join("?" for _ in cols)
            values = [row[c] for c in cols]
            cur.execute(
                f'INSERT INTO "{table_name}" VALUES ({placeholders})', values
            )

    conn.commit()
    return conn
