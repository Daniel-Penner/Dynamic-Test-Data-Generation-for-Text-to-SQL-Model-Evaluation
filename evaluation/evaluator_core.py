# evaluator_core.py
import sqlite3
import time
from typing import List, Dict, Tuple, Union

SQLITE_TIMEOUT_MS = 1000  # 1 second timeout


# -------------------------------
# Helpers
# -------------------------------
def rows_equal_value_only(gold_rows, pred_rows) -> bool:
    if len(gold_rows) != len(pred_rows):
        return False

    gold_tuples = [tuple(r.values()) for r in gold_rows]
    pred_tuples = [tuple(r.values()) for r in pred_rows]
    return gold_tuples == pred_tuples


def normalize_rows(rows):
    """
    Convert row tuples/dicts into a canonical list of dicts.
    (You probably don't need this if you always use sqlite3.Row + dict())
    """
    if isinstance(rows, dict):
        raise ValueError("normalize_rows should not receive error dicts.")
    if isinstance(rows, str):
        raise ValueError("normalize_rows received a string; this is a bug.")

    if rows is None:
        return []

    out = []
    for r in rows:
        if isinstance(r, dict):
            out.append(r)
        else:
            raise ValueError("Rows must be dicts. Ensure row_factory=sqlite3.Row.")
    return out


# -------------------------------
# Core SQLite execution
# -------------------------------
def _execute_with_timeout(conn: sqlite3.Connection, sql: str) -> Union[List[Dict], Dict]:
    """
    Internal helper: executes SQL on an existing connection with a timeout.
    Returns:
      - list[dict] on success
      - {"error": "..."} on failure / timeout
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


def execute_sqlite_query(db_path: str, sql: str) -> Union[List[Dict], Dict]:
    """
    Open a fresh connection to `db_path`, run the query with timeout,
    then close the connection.
    """
    try:
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
    except Exception as e:
        return {"error": str(e)}

    try:
        return _execute_with_timeout(conn, sql)
    finally:
        conn.close()


def execute_sqlite_query_conn(conn: sqlite3.Connection, sql: str) -> Union[List[Dict], Dict]:
    """
    Run SQL on an existing SQLite connection with timeout.
    Returns list[dict] or {"error": "..."}.
    """
    return _execute_with_timeout(conn, sql)


# -------------------------------
# Compare row sets for EX, F1, VES
# -------------------------------
def compare_results(pred, gold) -> Tuple[int, float, float]:
    """
    Return (EX, F1, VES)
    Using value-only comparison so column-name differences don't matter.
    Errors produce EX=0, F1=0, VES=0.

    - pred, gold: either list[dict] of rows, or {"error": "..."}.
    """

    # If either side is an error dict or not a list, treat as full failure
    if not isinstance(pred, list) or not isinstance(gold, list):
        return 0, 0.0, 0.0

    # Convert rows to tuples of values, ignoring column names
    gold_set = {tuple(row.values()) for row in gold}
    pred_set = {tuple(row.values()) for row in pred}

    # EX (exact match of sets)
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
