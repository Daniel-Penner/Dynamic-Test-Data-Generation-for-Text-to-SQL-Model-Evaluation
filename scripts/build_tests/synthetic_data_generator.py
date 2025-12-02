# scripts/build_tests/synthetic_data_generator.py

from __future__ import annotations

import random
import string
import re
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
    sql: str,
    order_info: Optional[Any] = None,
    n_rows_per_table: int = 40,
    scenario: Optional[Dict[str, Any]] = None,
    match_ratio: float = 0.25,
) -> Dict[str, pd.DataFrame]:

    # =============================================================
    # Normalize input: constraints comes as {required_equals,...}
    # or as a Constraints object
    # =============================================================
    if not isinstance(constraints, Constraints):
        # Build per-table constraint layout
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

        # required_equals
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

        # join keys come as 4-tuples: (t1,c1,t2,c2)
        for (t1, c1, t2, c2) in constraints.get("join_keys", set()):
            per_table[t1]["join_keys"].add((c1, t2, c2))
            per_table[t2]["join_keys"].add((c2, t1, c1))

        constraints = Constraints(per_table)

    c = constraints

    used_tables: set[str] = set()
    if sql:
        # Normalize whitespace and strip backticks so `frpm` → frpm
        s = " ".join(sql.replace("\n", " ").split())
        s = s.replace("`", "")

        # FROM <table>
        for m in re.finditer(r"\bFROM\s+([a-zA-Z0-9_]+)", s, re.IGNORECASE):
            used_tables.add(m.group(1).lower())

        # JOIN <table>
        for m in re.finditer(r"\bJOIN\s+([a-zA-Z0-9_]+)", s, re.IGNORECASE):
            used_tables.add(m.group(1).lower())

    # If parsing somehow failed, fall back to "all tables"
    if not used_tables:
        used_tables = set(schema_map.keys())

    # =============================================================
    # Extract direct required-equals for easy lookup
    # =============================================================
    required_eq_by_table: Dict[str, Dict[str, Any]] = {}
    for (tbl, col), val in c.required_equals.items():
        required_eq_by_table.setdefault(tbl, {})[col] = val

    # =============================================================
    # Resolve all join edges into canonical pairs
    # =============================================================
    join_edges = set()
    for table, join_set in c.join_keys_by_table.items():
        for (col, other_table, other_col) in join_set:
            edge = tuple(sorted([(table, col), (other_table, other_col)]))
            if len(edge) == 2:
                (t1, c1), (t2, c2) = edge
                join_edges.add((t1, c1, t2, c2))

    # =============================================================
    # STEP 1: Identify which table has WHERE constraints
    #         and create forced join rows to guarantee matches
    # =============================================================
    forced_join_values: Dict[Tuple[str, str], Any] = {}

    for (t1, c1, t2, c2) in join_edges:
        left_has_constraints = bool(required_eq_by_table.get(t1, {}))

        if left_has_constraints:
            shared_val = _random_string("join")
            forced_join_values[(t1, c1)] = shared_val
            forced_join_values[(t2, c2)] = shared_val

        right_has_constraints = bool(required_eq_by_table.get(t2, {}))
        if right_has_constraints:
            shared_val = _random_string("join")
            forced_join_values[(t1, c1)] = shared_val
            forced_join_values[(t2, c2)] = shared_val

    # =============================================================
    # STEP 2: Build normal global join propagation map
    #         (for rows NOT used in forced join pairs)
    # =============================================================
    global_join_values: Dict[Tuple[str, str], Any] = {}

    for (t1, c1, t2, c2) in join_edges:
        k1 = (t1, c1)
        k2 = (t2, c2)

        if k1 in forced_join_values:
            val = forced_join_values[k1]
            global_join_values[k2] = val
            continue

        if k2 in forced_join_values:
            val = forced_join_values[k2]
            global_join_values[k1] = val
            continue

        # Otherwise: free random join cluster
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
    # STEP 3: Generate data table-by-table
    # =============================================================
    dataset: Dict[str, pd.DataFrame] = {}

    for table_name, columns in schema_map.items():
        # Skip tables not referenced in the SQL query
        if table_name.lower() not in used_tables:
            continue

        table_required_eq = required_eq_by_table.get(table_name, {})
        rows: List[Dict[str, Any]] = []

        # ---------------------------------------------------------
        # 3A: Guaranteed WHERE-matching + join-matching rows
        # ---------------------------------------------------------
        if table_required_eq or any((table_name, col) in forced_join_values for col in columns):

            forced_row = {}
            for col, dtype in columns.items():
                # WHERE constraint
                if col in table_required_eq:
                    forced_row[col] = _coerce_value_for_dtype(table_required_eq[col], dtype)
                    continue

                # Forced join key
                if (table_name, col) in forced_join_values:
                    forced_row[col] = _coerce_value_for_dtype(forced_join_values[(table_name, col)], dtype)
                    continue

                # Normal join propagation
                jk = (table_name, col)
                if jk in global_join_values:
                    forced_row[col] = _coerce_value_for_dtype(global_join_values[jk], dtype)
                    continue

                # Random fallback
                t = dtype.lower()
                if "char" in t or "text" in t:
                    forced_row[col] = _random_string("txt")
                elif "int" in t or "float" in t or "real" in t or "double" in t:
                    forced_row[col] = _random_numeric()
                elif "date" in t:
                    forced_row[col] = "2005-01-01"
                else:
                    forced_row[col] = _random_string("v")

            rows.append(forced_row)

        # ---------------------------------------------------------
        # 3B: Normal rows (matching + non-matching)
        # ---------------------------------------------------------
        match_count = max(1, int(n_rows_per_table * match_ratio))

        for _ in range(match_count):
            row = {}
            for col, dtype in columns.items():

                if col in table_required_eq:
                    row[col] = _coerce_value_for_dtype(table_required_eq[col], dtype)
                    continue

                if (table_name, col) in global_join_values:
                    row[col] = _coerce_value_for_dtype(global_join_values[(table_name, col)], dtype)
                    continue

                t = dtype.lower()
                if "char" in t or "text" in t:
                    row[col] = _random_string("match")
                elif "int" in t or "float" in t or "real" in t or "double" in t:
                    row[col] = _random_numeric()
                elif "date" in t:
                    row[col] = "2005-01-01"
                else:
                    row[col] = _random_string("v")

            rows.append(row)

        for _ in range(n_rows_per_table - match_count):
            row = {}
            for col, dtype in columns.items():

                jk = (table_name, col)
                if jk in global_join_values:
                    row[col] = _coerce_value_for_dtype(global_join_values[jk], dtype)
                    continue

                t = dtype.lower()
                if "char" in t or "text" in t:
                    row[col] = _random_string("s")
                elif "int" in t or "float" in t or "real" in t or "double" in t:
                    row[col] = _random_numeric()
                elif "date" in t:
                    row[col] = "2005-01-01"
                else:
                    row[col] = _random_string("v")

            rows.append(row)

        df = pd.DataFrame(rows)

        # Apply scenario mutation
        if scenario and scenario.get("inject"):
            df = scenario["inject"](df)

        # Apply NOT NULL constraints
        for (tbl, col) in c.required_not_null:
            if tbl == table_name and col in df.columns:
                df[col] = df[col].fillna(_random_numeric())

        dataset[table_name] = df

    return dataset