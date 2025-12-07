# scripts/build_tests/witness_planner.py

from typing import Dict, Tuple, List, Any
from datetime import date, timedelta
import re


def _make_greater_than(val):
    if isinstance(val, str) and re.match(r"\d{4}-\d{2}-\d{2}", val):
        y, m, d = map(int, val.split("-"))
        return date(y, m, d) + timedelta(days=1)

    try:
        return float(val) + 1
    except Exception:
        return val


def _make_less_than(val):
    if isinstance(val, str) and re.match(r"\d{4}-\d{2}-\d{2}", val):
        y, m, d = map(int, val.split("-"))
        return date(y, m, d) - timedelta(days=1)

    try:
        return float(val) - 1
    except Exception:
        return val


def plan_witness_assignments(constraints, k: int = 3) -> List[Dict[Tuple[str, str], Any]]:
    """
    Build k witness assignments that *should* satisfy the query constraints.

    `constraints` is a Constraints object defined in synthetic_data_generator.py.
    """
    witnesses: List[Dict[Tuple[str, str], Any]] = []

    for i in range(k):
        w: Dict[Tuple[str, str], Any] = {}

        # ----------------------------------------
        # WHERE col = value
        # ----------------------------------------
        for (tbl, col), val in getattr(constraints, "required_equals", {}).items():
            w[(tbl, col)] = val

        # ----------------------------------------
        # DO NOT directly assign required_not_null here
        # let enforce_winner_constraints() set safe values (e.g., 1)
        # ----------------------------------------

        # ----------------------------------------
        # SCALAR SUBQUERIES: outer.col = inner.col
        # ----------------------------------------
        for (t1, c1, t2, c2) in getattr(constraints, "scalar_subqueries", set()):
            shared = f"SCALAR_{i}"
            w[(t1, c1)] = shared
            w[(t2, c2)] = shared

        # ----------------------------------------
        # JOIN KEYS
        # constraints.join_keys_by_table: {tbl: {(col, other_tbl, other_col), ...}}
        # ----------------------------------------
        for tbl, joins in getattr(constraints, "join_keys_by_table", {}).items():
            for col, other_tbl, other_col in joins:
                shared = f"JOINKEY_{i}"
                w[(tbl, col)] = shared
                w[(other_tbl, other_col)] = shared

        # ----------------------------------------
        # RANGE constraints
        # ----------------------------------------
        for (tbl, col, op, val) in getattr(constraints, "range_constraints", []):
            if op in (">", ">="):
                w[(tbl, col)] = _make_greater_than(val)
            elif op in ("<", "<="):
                w[(tbl, col)] = _make_less_than(val)

        # NOTE: we deliberately skip order_by / having here.
        # order_by is handled by order_info in the generator,
        # and HAVING is too complex to fully model here.

        witnesses.append(w)

    return witnesses
