# scripts/build_tests/synthetic_data_generator.py

from __future__ import annotations

import random
import string
from dataclasses import dataclass, field
from typing import Any, Dict, Tuple, Set, List, Optional

import pandas as pd


# =====================================================================
# Helper random generators
# =====================================================================

def _random_string(prefix: str, length: int = 6) -> str:
    suffix = "".join(random.choices(string.digits, k=length))
    return f"{prefix}_{suffix}"


def _random_numeric() -> float:
    return float(random.randint(1, 1000))


def _coerce_value_for_dtype(val: Any, dtype: str) -> Any:
    if val is None:
        return None

    t = dtype.lower()

    if "int" in t:
        try:
            return int(val)
        except:
            try:
                return int(float(val))
            except:
                return 0

    if "real" in t or "float" in t or "double" in t:
        try:
            return float(val)
        except:
            return 0.0

    if "char" in t or "text" in t or "varchar" in t:
        return str(val)

    if "date" in t:
        return str(val)

    return val


# =====================================================================
# Constraints wrapper
# =====================================================================

@dataclass
class Constraints:
    per_table: Dict[str, Dict[str, Any]]

    required_equals: Dict[Tuple[str, str], Any] = field(default_factory=dict)
    required_not_equals: Dict[Tuple[str, str], Any] = field(default_factory=dict)
    required_is_null: Set[Tuple[str, str]] = field(default_factory=set)
    required_not_null: Set[Tuple[str, str]] = field(default_factory=set)

    join_keys_by_table: Dict[str, Set[Tuple[str, str, str]]] = field(default_factory=dict)

    required_join_keys: Set[str] = field(default_factory=set)

    def __post_init__(self):
        for table, info in self.per_table.items():
            rqeq = info.get("required_equals", {}) or {}
            rqneq = info.get("required_not_equals", {}) or {}
            isnull = info.get("required_is_null", set()) or set()
            notnull = info.get("required_not_null", set()) or set()
            join_keys = info.get("join_keys", set()) or set()

            for col, val in rqeq.items():
                self.required_equals[(table, col)] = val

            for col, val in rqneq.items():
                self.required_not_equals[(table, col)] = val

            for col in isnull:
                self.required_is_null.add((table, col))

            for col in notnull:
                self.required_not_null.add((table, col))

            self.join_keys_by_table[table] = set(join_keys)

            for col, other_table, other_col in join_keys:
                self.required_join_keys.add(col)
                self.required_join_keys.add(other_col)


# =====================================================================
# Main generator
# =====================================================================

def generate_synthetic_dataset(
    schema_map: Dict[str, Dict[str, str]],
    constraints: Any,
    order_info: Optional[Any] = None,
    n_rows_per_table: int = 40,
    scenario: Optional[Dict[str, Any]] = None,
    match_ratio: float = 0.25,
) -> Dict[str, pd.DataFrame]:

    print("\n[DEBUG constraints input]:", constraints, type(constraints))

    # =============================================================
    # FIX: Normalize GLOBAL constraint dict → per-table constraints
    # =============================================================
    if isinstance(constraints, Constraints):
        c = constraints

    elif isinstance(constraints, dict):

        # Build an empty per-table structure
        per_table = {
            tbl: {
                "required_equals": {},
                "required_not_equals": {},
                "required_is_null": set(),
                "required_not_null": set(),
                "join_keys": set(),
            }
            for tbl in schema_map.keys()
        }

        # required_equals: { (table, col): val }
        for (tbl, col), val in constraints.get("required_equals", {}).items():
            per_table[tbl]["required_equals"][col] = val

        # required_not_equals
        for (tbl, col), val in constraints.get("required_not_equals", {}).items():
            per_table[tbl]["required_not_equals"][col] = val

        # is null
        for (tbl, col) in constraints.get("required_is_null", set()):
            per_table[tbl]["required_is_null"].add(col)

        # not null
        for (tbl, col) in constraints.get("required_not_null", set()):
            per_table[tbl]["required_not_null"].add(col)

        # join keys: stored as (table, col, other_table, other_col)
        for item in constraints.get("join_keys", set()):
            table, col, other_table, other_col = item
            per_table[table]["join_keys"].add((col, other_table, other_col))
            per_table[other_table]["join_keys"].add((other_col, table, col))

        c = Constraints(per_table)

    else:
        raise TypeError(
            f"constraints must be dict or Constraints, got {type(constraints)}"
        )

    # =============================================================
    # Build lookup maps
    # =============================================================
    dataset: Dict[str, pd.DataFrame] = {}

    required_eq_by_table: Dict[str, Dict[str, Any]] = {}
    for (tbl, col), val in c.required_equals.items():
        required_eq_by_table.setdefault(tbl, {})[col] = val

    # join key → shared value
    global_join_values: Dict[Tuple[str, str], Any] = {}

    for table, join_set in c.join_keys_by_table.items():
        for col, other_table, other_col in join_set:
            k1 = (table, col)
            k2 = (other_table, other_col)

            if k1 in global_join_values:
                v = global_join_values[k1]
                global_join_values[k2] = v
            elif k2 in global_join_values:
                v = global_join_values[k2]
                global_join_values[k1] = v
            else:
                v = _random_string("key")
                global_join_values[k1] = v
                global_join_values[k2] = v

    # =============================================================
    # Generate tables
    # =============================================================
    for table_name, columns in schema_map.items():

        rows: List[Dict[str, Any]] = []

        table_required_eq = required_eq_by_table.get(table_name, {})
        match_count = max(1, int(n_rows_per_table * match_ratio))

        # ---------------------------------------------------------
        # Matching rows
        # ---------------------------------------------------------
        for _ in range(match_count):
            row = {}
            for col, dtype in columns.items():

                # WHERE equality
                if col in table_required_eq:
                    raw_val = table_required_eq[col]
                    row[col] = _coerce_value_for_dtype(raw_val, dtype)
                    continue

                # join keys
                jk = (table_name, col)
                if jk in global_join_values:
                    row[col] = _coerce_value_for_dtype(global_join_values[jk], dtype)
                    continue

                # random fallback
                t = dtype.lower()
                if "char" in t or "text" in t or "varchar" in t:
                    row[col] = _random_string("match")
                elif "date" in t:
                    row[col] = "2005-01-01"
                elif "int" in t or "real" in t or "float" in t or "double" in t:
                    row[col] = _random_numeric()
                else:
                    row[col] = _random_string("v")

            rows.append(row)

        # ---------------------------------------------------------
        # Non-matching rows
        # ---------------------------------------------------------
        for _ in range(n_rows_per_table - match_count):
            row = {}
            for col, dtype in columns.items():

                jk = (table_name, col)
                if jk in global_join_values:
                    row[col] = _coerce_value_for_dtype(global_join_values[jk], dtype)
                    continue

                t = dtype.lower()
                if "char" in t or "text" in t or "varchar" in t:
                    row[col] = _random_string("s")
                elif "date" in t:
                    row[col] = "2005-01-01"
                elif "int" in t or "real" in t or "float" in t or "double":
                    row[col] = _random_numeric()
                else:
                    row[col] = _random_string("v")

            rows.append(row)

        df = pd.DataFrame(rows)

        # Scenario mutations
        if scenario and scenario.get("inject"):
            df = scenario["inject"](df)

        # NOT NULL enforcement
        for (tbl, col) in c.required_not_null:
            if tbl == table_name and col in df.columns:
                df[col] = df[col].fillna(_random_numeric())

        dataset[table_name] = df

    return dataset
