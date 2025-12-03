# evaluator_core.py
import sqlite3
import json
import math

def rows_equal_value_only(gold_rows, pred_rows):
    if len(gold_rows) != len(pred_rows):
        return False
    
    # Convert each row to tuple(values sorted by key) or
    # preserve order from SQLite description if needed
    gold_tuples = [tuple(r.values()) for r in gold_rows]
    pred_tuples = [tuple(r.values()) for r in pred_rows]
    
    return gold_tuples == pred_tuples


# -------------------------------
# Normalize SQLite row outputs
# -------------------------------
def normalize_rows(rows):
    if isinstance(rows, dict):
        raise ValueError("normalize_rows should not receive error dicts.")
    if isinstance(rows, str):
        raise ValueError("normalize_rows received a string (bug: execute_sqlite_query must not return strings)")

    """Convert row tuples/dicts into a canonical list of dicts."""
    if rows is None:
        return []

    out = []
    for r in rows:
        if isinstance(r, dict):
            out.append(r)
        else:
            # assume tuple → convert to dict using cursor description
            raise ValueError("Rows must be dicts. Ensure row_factory=sqlite3.Row.")
    return out
import sqlite3, time

SQLITE_TIMEOUT_MS = 1000   # 1 second timeout

def execute_sqlite_query(db_path, sql):
    try:
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
    except Exception as e:
        return {"error": str(e)}

    start = time.time()

    # Progress handler called every 1000 SQLite VM instructions
    def timeout_checker():
        if (time.time() - start) * 1000 >= SQLITE_TIMEOUT_MS:
            return 1   # Abort query
        return 0

    conn.set_progress_handler(timeout_checker, 1000)

    try:
        cur = conn.cursor()
        cur.execute(sql)
        rows = [dict(row) for row in cur.fetchall()]
        return rows
    except Exception as e:
        return {"error": str(e)}
    finally:
        conn.set_progress_handler(None, 0)
        conn.close()

SQLITE_TIMEOUT_MS = 1000  # 1 second

def execute_sqlite_query_conn(conn: sqlite3.Connection, sql: str):
    """
    Execute SQL on an existing in-memory SQLite conn with timeout.
    Returns:
      - list of row dicts on success
      - {"error": "..."} on failure or timeout
    """

    start = time.time()

    def timeout_checker():
        elapsed_ms = (time.time() - start) * 1000
        if elapsed_ms >= SQLITE_TIMEOUT_MS:
            return 1  # abort query
        return 0

    conn.set_progress_handler(timeout_checker, 1000)

    try:
        cur = conn.cursor()
        cur.execute(sql)
        rows = [dict(row) for row in cur.fetchall()]
        return rows
    except Exception as e:
        return {"error": str(e)}
    finally:
        conn.set_progress_handler(None, 0)

# -------------------------------
# Compare row sets for EX, F1, VES
# -------------------------------
def compare_results(pred, gold):
    """
    Return (EX, F1, VES)
    Using value-only tuple comparison so column-name differences don't matter.
    Errors produce EX=0, F1=0, VES=0.
    """

    # If error -> EX = 0
    if not isinstance(pred, list) or not isinstance(gold, list):
        return 0, 0.0, 0.0

    # Convert rows to tuples of values, ignoring column names
    gold_set = {tuple(row.values()) for row in gold}
    pred_set = {tuple(row.values()) for row in pred}

    # EX
    ex = int(pred_set == gold_set)

    # F1
    tp = len(gold_set & pred_set)
    fp = len(pred_set - gold_set)
    fn = len(gold_set - pred_set)

    precision = tp / (tp + fp) if tp + fp > 0 else 0.0
    recall = tp / (tp + fn) if tp + fn > 0 else 0.0
    f1 = (2 * precision * recall) / (precision + recall) if precision + recall else 0.0

    # VES = (tp - fp - fn) / |gold|
    ves = (tp - fp - fn) / len(gold_set) if gold_set else 0.0

    return ex, f1, ves

# -------------------------------
# Execute SQL with debug output
# -------------------------------
def execute_sql(conn, sql, label=""):
    try:
        cur = conn.cursor()
        cur.execute(sql)
        rows = [dict(row) for row in cur.fetchall()]
        return True, rows, None
    except Exception as e:
        return False, None, str(e)
