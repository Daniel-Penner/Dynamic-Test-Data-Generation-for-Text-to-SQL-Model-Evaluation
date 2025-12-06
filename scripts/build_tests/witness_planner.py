# scripts/build_tests/witness_planner.py

from typing import Dict, Tuple, List, Any
import uuid

from datetime import date, timedelta
import re

def _make_greater_than(val, col):
    # Date literal
    if isinstance(val, str) and re.match(r"\d{4}-\d{2}-\d{2}", val):
        y, m, d = map(int, val.split("-"))
        return date(y, m, d) + timedelta(days=1)

    # Numeric
    try:
        return float(val) + 1
    except:
        return val


def _make_less_than(val, col):
    if isinstance(val, str) and re.match(r"\d{4}-\d{2}-\d{2}", val):
        y, m, d = map(int, val.split("-"))
        return date(y, m, d) - timedelta(days=1)

    try:
        return float(val) - 1
    except:
        return val


def plan_witness_assignments(constraints, k: int = 3):
    witnesses = []

    for i in range(k):
        w = {}

        # WHERE col = value
        for (tbl, col), val in constraints.required_equals.items():
            w[(tbl, col)] = val

        # NOT NULL columns
        for (tbl, col) in constraints.required_not_null:
            w[(tbl, col)] = f"{col}_W{i}"

        # JOIN keys — deterministic per witness
        for tbl, join_defs in constraints.join_keys_by_table.items():
            for col, other_tbl, other_col in join_defs:
                shared = f"JOINKEY_{i}"
                w[(tbl, col)] = shared
                w[(other_tbl, other_col)] = shared

        # RANGE constraints must override everything
        for (tbl, col, op, val) in constraints.range_constraints:
            if op in (">", ">="):
                w[(tbl, col)] = _make_greater_than(val, col)
            elif op in ("<", "<="):
                w[(tbl, col)] = _make_less_than(val, col)

        witnesses.append(w)

    return witnesses