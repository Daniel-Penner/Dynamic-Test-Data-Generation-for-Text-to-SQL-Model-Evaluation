# scripts/build_tests/synthetic_data_generator.py

from __future__ import annotations

import random
import string
import re
from dataclasses import dataclass, field
from typing import Any, Dict, Tuple, Set, List, Optional
from scripts.build_tests.witness_planner import plan_witness_assignments
from datetime import date, timedelta


import pandas as pd
from collections.abc import Mapping

#ALGORITHMIC GENERATION OF SYNTHETIC DATA

import re


def _enforce_string_equalities(dataset, constraints):
    for (tbl, col, val) in getattr(constraints, "equal_constraints", []):
        if tbl not in dataset:
            continue

        df = dataset[tbl]
        if col not in df.columns:
            continue

        if (df[col] == val).any():
            continue

        df.loc[df.index[0], col] = val


def _sanitize_division_denominators(dataset, sql):
    if "/" not in sql:
        return

    for tbl, df in dataset.items():
        for col in df.columns:
            name = col.lower()

            if any(k in name for k in ("enrollment", "count", "total", "num", "number")):
                if pd.api.types.is_numeric_dtype(df[col]):
                    df[col] = df[col].apply(
                        lambda x: 1 if (x is None or x == 0) else x
                    )

def safe_numeric(val, fallback=(1, 100)):
    if isinstance(val, (int, float)):
        return val
    return random.randint(*fallback)

def coerce_numeric(val, series):
    if isinstance(val, str):
        v = val.strip().strip(";").strip("'").strip('"')
        if v.lower().startswith("("):
            return None
        try:
            return float(v) if "." in v else int(v)
        except Exception:
            return None

    if isinstance(val, (int, float)):
        return val

    return None


def is_query_satisfiable(constraints: Constraints) -> bool:
    by_col = {}

    for (tbl, col), v in constraints.required_equals.items():
        by_col.setdefault((tbl, col), []).append(("=", v))

    for (tbl, col, op, v) in constraints.range_constraints:
        by_col.setdefault((tbl, col), []).append((op, v))

    for (tbl, col), ops in by_col.items():
        lo = -float("inf")
        hi = float("inf")
        eq = None

        for op, v in ops:
            if op == "=":
                eq = v
            elif op == ">":
                lo = max(lo, safe_numeric(v) + 1)
            elif op == ">=":
                lo = max(lo, v)
            elif op == "<":
                hi = min(hi, safe_numeric(v) - 1)
            elif op == "<=":
                hi = min(hi, v)
            elif op == "BETWEEN":
                lo = max(lo, v[0])
                hi = min(hi, v[1])

        has_numeric_ops = any(op in {">", ">=", "<", "<=", "BETWEEN"} for op, _ in ops)
        if not has_numeric_ops:
            continue

        if eq is not None:
            if isinstance(eq, (int, float)):
                if not (lo <= eq <= hi):
                    return False
        if lo > hi:
            return False

    if getattr(constraints, "set_op", None) == "INTERSECT":
        for col in constraints.projected_cols:
            vals = []
            for (t, c), v in constraints.required_equals.items():
                if c == col:
                    vals.append(v)
            if len(vals) > 1 and len(set(vals)) > 1:
                return False

    return True

def canonical_join_value(col: str, idx: int = 0) -> str:
    return f"JOINVAL_{idx}"

def infer_numeric_columns_from_sql(sql: str) -> set[str]:
    sql = sql.lower()
    numeric_cols = set()

    for fn in ("avg", "sum", "min", "max", "count"):
        for m in re.finditer(
            rf"{fn}\s*\(\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*\)",
            sql
        ):
            numeric_cols.add(m.group(1))

    for m in re.finditer(
        r"([a-zA-Z_][a-zA-Z0-9_]*)\s*(>=|<=|>|<|=)\s*[0-9]",
        sql
    ):
        numeric_cols.add(m.group(1))

    for m in re.finditer(
        r"([a-zA-Z_][a-zA-Z0-9_]*)\s+between\s+[0-9]+\s+and\s+[0-9]+",
        sql
    ):
        numeric_cols.add(m.group(1))

    return {c.lower() for c in numeric_cols}


def _infer_dtype(col_name, example_values=None):

    name = col_name.lower()
    tokens = [t for t in re.split(r"[^a-z0-9]+", name) if t]

    if "grade" in tokens:
        return "int"

    numeric_tokens = {"count", "num", "number", "total", "enrollment", "sum"}
    if any(t in numeric_tokens for t in tokens):
        return "int"

    if "%" in col_name or "percent" in tokens or "ratio" in tokens or "rate" in tokens:
        return "float"

    if "latitude" in tokens or "longitude" in tokens:
        return "float"

    if "date" in tokens:
        return "date"

    if "year" in tokens:
        return "int"

    return "text"


def _deterministic_value(col: str, dtype: str, row_idx: int | None):
    if dtype in ("float", "real", "double"):
        return round(random.uniform(1, 5000), 3)

    if dtype in ("int", "integer"):
        return random.randint(1, 5000)
    
    if dtype == "date":
        if row_idx is None:
            return date(
                random.randint(1970, 2030),
                random.randint(1, 12),
                random.randint(1, 28),
            )

        base = date(1995, 1, 1)
        return base + timedelta(days=row_idx * 400)

    if row_idx is None:
        row_idx = random.randint(1, 99999)
    suffix = f"{row_idx:05d}"
    return f"{col}_{suffix}"

def enforce_winner_constraints(
    dataset: Dict[str, pd.DataFrame],
    constraints: Constraints,
    winner_idx: int = 0,
    allow_joins: bool = True,
    allow_scalar_subqueries: bool = True,
    order_winner_idx: Optional[Dict[str, int]] = None,
):

    c = constraints

    for (tbl, col), val in c.required_equals.items():
        if tbl not in dataset or col not in dataset[tbl].columns:
            continue

        dataset[tbl].loc[winner_idx, col] = _coerce_value_for_dtype(
            val, str(dataset[tbl][col].dtype)
        )

        for (t_join, join_defs) in c.join_keys_by_table.items():
            if t_join != tbl:
                continue

            for (c1, other_tbl, c2) in join_defs:
                if c1 != col:
                    continue 

                if (
                    other_tbl in dataset
                    and c2 in dataset[other_tbl].columns
                ):
                    dataset[other_tbl].loc[winner_idx, c2] = dataset[tbl].loc[winner_idx, col]

    for (tbl, col, pat) in getattr(c, "like_constraints", []):
        if tbl in dataset and col in dataset[tbl].columns:
            dataset[tbl].loc[winner_idx, col] = f"xxx{pat.strip('%')}yyy"

    for (tbl, col, op, val) in getattr(c, "range_constraints", []):
        if tbl not in dataset or col not in dataset[tbl].columns:
            continue

        df = dataset[tbl]

        num_val = coerce_numeric(val, df[col])

        if op == "=":
            df.loc[winner_idx, col] = num_val if num_val is not None else val

        elif op == ">":
            if num_val is not None:
                df.loc[winner_idx, col] = num_val + 1

        elif op == ">=":
            if num_val is not None:
                df.loc[winner_idx, col] = num_val

        elif op == "<":
            if num_val is not None:
                df.loc[winner_idx, col] = num_val - 1

        elif op == "<=":
            if num_val is not None:
                df.loc[winner_idx, col] = num_val

        elif op == "BETWEEN":
            lo, hi = val
            lo_n = coerce_numeric(lo, df[col])
            hi_n = coerce_numeric(hi, df[col])
            if lo_n is not None and hi_n is not None:
                df.loc[winner_idx, col] = (lo_n + hi_n) // 2


    for (tbl, col) in c.required_not_null:
        if tbl in dataset and col in dataset[tbl].columns:
            df = dataset[tbl]
            if winner_idx < len(df):
                if pd.isna(dataset[tbl].loc[winner_idx, col]):
                    dataset[tbl].loc[winner_idx, col] = 1

    if allow_joins:
        for tbl, join_defs in c.join_keys_by_table.items():
            if tbl not in dataset:
                continue
            df = dataset[tbl]
            if winner_idx >= len(df):
                continue

            for col, other_tbl, other_col in join_defs:
                if (
                    other_tbl not in dataset
                    or col not in df.columns
                    or other_col not in dataset[other_tbl].columns
                ):
                    continue

                other_df = dataset[other_tbl]
                if winner_idx >= len(other_df):
                    continue

                val = df.loc[winner_idx, col]
                if pd.isna(val):
                    val = other_df.loc[winner_idx, other_col]

                if pd.isna(val):
                    val = f"WINNER_JOIN_{tbl}_{col}"

                df.loc[winner_idx, col] = val
                other_df.loc[winner_idx, other_col] = val

    from pandas.api.types import is_numeric_dtype

    for (tbl, col), val in c.required_equals.items():
        if tbl not in dataset or col not in dataset[tbl].columns:
            continue

        series = dataset[tbl][col]
        num = coerce_numeric(val, series)

        if num is not None:
            dataset[tbl].loc[winner_idx, col] = num
        else:
            dataset[tbl].loc[winner_idx, col] = str(val).strip(";").strip("'")

    for (tbl, col, pat) in getattr(c, "like_constraints", []):
        if tbl in dataset and col in dataset[tbl].columns:
            dataset[tbl].loc[winner_idx, col] = f"xxx{pat.strip('%')}yyy"

    for (tbl, col, op, val) in c.range_constraints:
        if tbl not in dataset or col not in dataset[tbl].columns:
            continue

        series = dataset[tbl][col]
        num = coerce_numeric(val, series)

        if num is None:
            continue

        if op == ">":
            dataset[tbl].loc[winner_idx, col] = num + 1
        elif op == ">=":
            dataset[tbl].loc[winner_idx, col] = num
        elif op == "<":
            dataset[tbl].loc[winner_idx, col] = num - 1
        elif op == "<=":
            dataset[tbl].loc[winner_idx, col] = num

    if allow_scalar_subqueries and getattr(constraints, "scalar_subqueries", None):
        for (outer_tbl, outer_col, inner_tbl, inner_col) in c.scalar_subqueries:
            if outer_tbl not in dataset or inner_tbl not in dataset:
                continue

            odf = dataset[outer_tbl]
            idf = dataset[inner_tbl]

            if (
                outer_col not in odf.columns
                or inner_col not in idf.columns
                or winner_idx >= len(odf)
                or winner_idx >= len(idf)
            ):
                continue

            outer_val = odf.loc[winner_idx, outer_col]
            inner_val = idf.loc[winner_idx, inner_col]

            if pd.notna(outer_val):
                shared = outer_val
            elif pd.notna(inner_val):
                shared = inner_val
            else:
                shared = random.randint(500, 1500)

            odf.loc[winner_idx, outer_col] = shared
            idf.loc[winner_idx, inner_col] = shared



def _random_string(prefix: str, length: int = 6) -> str:
    suffix = "".join(random.choices(string.digits, k=length))
    return f"{prefix}_{suffix}"

def rnd(col, dtype):
    t = dtype.lower()
    if "int" in t:
        return random.randint(1, 100)
    if "float" in t or "real" in t:
        return random.random() * 10
    if "date" in t:
        return f"19{random.randint(70,99)}-{random.randint(1,12):02d}-{random.randint(1,28):02d}"
    if "char" in t or "text" in t:
        return _random_string("txt")
    return _random_string("v")


def _coerce_value_for_dtype(val: Any, dtype: str) -> Any:
    if isinstance(val, str) and val.startswith("__LIKE_MATCH_"):
        core = val.replace("__LIKE_MATCH_", "").replace("__", "")
        return f"xxx{core}yyy"
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

def get_like_match(table, col, constraints):
    for (t, c, val) in getattr(constraints, "like_constraints", []):
        if t == table and c == col:
            return f"__LIKE_MATCH_{val.replace('%','')}__"
    return None

def get_required_value(
    table_name: str,
    col: str,
    dtype: str,
    witnesses: list[dict],
    wi: int,
):
    key = (table_name, col)
    if wi < len(witnesses) and key in witnesses[wi]:
        v = witnesses[wi][key]
        if isinstance(v, str) and "%" in v:
            return f"__LIKE_MATCH_{v.replace('%', '')}__"
        return _coerce_value_for_dtype(v, dtype)
    return None


def generate_join_keys(n: int, prefix: str = "jk") -> list[str]:
    return [f"{prefix}_{i:05d}" for i in range(n)]

def _enforce_or_branch(dataset, schema_map, or_groups, alias_map=None):
    if not or_groups:
        return

    chosen = or_groups[0]  # deterministic choice

    pred_pattern = re.compile(
        r"(?:(?P<table>[a-zA-Z0-9_]+)\.)?(?P<column>[a-zA-Z0-9_`]+)\s*=\s*'?(?P<value>[^']+)'?",
        re.IGNORECASE,
    )

    for pred in chosen:
        m = pred_pattern.search(pred)
        if not m:
            continue

        tbl = m.group("table")
        col = m.group("column").strip("`")
        val = m.group("value")

        if tbl is None:
            matches = []
            for t, cols in schema_map.items():
                if col in cols:
                    matches.append(t)
            if len(matches) != 1:
                continue
            tbl = matches[0]
        else:
            tbl = tbl.lower()

        if tbl in dataset and col in dataset[tbl].columns:
            dataset[tbl].loc[0, col] = val

#Constraints

@dataclass
class Constraints:
    per_table: Dict[str, Dict[str, Any]]
    projected_cols: list[str] = field(default_factory=list)
    required_equals: Dict[Tuple[str, str], Any] = field(default_factory=dict)
    required_not_equals: Dict[Tuple[str, str], Any] = field(default_factory=dict)
    required_is_null: Set[Tuple[str, str]] = field(default_factory=set)
    required_not_null: Set[Tuple[str, str]] = field(default_factory=set)

    join_keys_by_table: Dict[str, Set[Tuple[str, str, str]]] = field(default_factory=dict)

    required_join_keys: Set[str] = field(default_factory=set)

    scalar_subqueries: Set[Tuple[str, str, str, str]] = field(default_factory=set)

    def __post_init__(self):
        self.range_constraints = []
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


#Generation process

def generate_synthetic_dataset(
    schema_map: Dict[str, Dict[str, str]],
    constraints: Any,
    sql: str,
    order_info: Optional[Any] = None,
    n_rows_per_table: int = 40,
    scenario: Optional[Dict[str, Any]] = None,
    match_ratio: float = 0.25,
) -> Dict[str, pd.DataFrame]:
    WINNER_IDX = 0
    order_winner_idx: Dict[str, int] = {}

    def is_numeric_dtype(dtype: str) -> bool:
        t = dtype.lower()
        return ("int" in t) or ("real" in t) or ("float" in t) or ("double" in t)

    is_deterministic = not scenario or ("inject" not in scenario or scenario["inject"] is None)

    original_examples = {
        tbl: list(cols.keys()) for tbl, cols in schema_map.items()
    }

    if isinstance(constraints, Constraints):
        c = constraints
    elif isinstance(constraints, Mapping):
        per_table = {
            tbl.lower(): {
                "required_equals": {},
                "required_not_equals": {},
                "required_is_null": set(),
                "required_not_null": set(),
                "join_keys": set(),
            }
            for tbl in schema_map.keys()
        }

        for (tbl, col), val in constraints.get("required_equals", {}).items():
            tbl_l = tbl.lower()
            if tbl_l in per_table:
                per_table[tbl_l]["required_equals"][col] = val

        for (tbl, col), val in constraints.get("required_not_equals", {}).items():
            tbl_l = tbl.lower()
            if tbl_l in per_table:
                per_table[tbl_l]["required_not_equals"][col] = val

        for (tbl, col) in constraints.get("required_is_null", set()):
            tbl_l = tbl.lower()
            if tbl_l in per_table:
                per_table[tbl_l]["required_is_null"].add(col)

        for (tbl, col) in constraints.get("required_not_null", set()):
            tbl_l = tbl.lower()
            if tbl_l in per_table:
                per_table[tbl_l]["required_not_null"].add(col)

        for (t1, c1, t2, c2) in constraints.get("join_keys", set()):
            t1_l = t1.lower()
            t2_l = t2.lower()
            if t1_l in per_table:
                per_table[t1_l]["join_keys"].add((c1, t2_l, c2))
            if t2_l in per_table:
                per_table[t2_l]["join_keys"].add((c2, t1_l, c1))

        range_constraints = constraints.get("range_constraints", [])
        scalar_subqueries = constraints.get("scalar_subqueries", set())

        constraints_obj = Constraints(per_table)
        constraints_obj.projected_cols = constraints.get("projected_cols", [])
        constraints_obj.range_constraints = list(range_constraints)
        constraints_obj.scalar_subqueries = set(scalar_subqueries)
        for (tbl, col), val in constraints_obj.required_equals.items():
            constraints_obj.range_constraints.append((tbl, col, "=", val))

        # Also propagate OR / offset info if present
        constraints_obj.or_groups = constraints.get("or_groups", [])
        constraints_obj.has_or = constraints.get("has_or", False)
        constraints_obj.offset = constraints.get("offset", None)

        c = constraints_obj
    else:
        raise TypeError(
            f"constraints must be a Constraints or mapping, got {type(constraints)}"
        )

    #Witness assignment for ensuring passing rows
    WITNESS_COUNT = 3
    witnesses = plan_witness_assignments(c, k=WITNESS_COUNT)
    winner = witnesses[0]

    WITNESS_ROW_INDEX = {tbl: 0 for tbl in schema_map}

    used_tables: set[str] = set()
    if sql:
        s = " ".join(sql.replace("\n", " ").split())
        s = s.replace("`", "")

        for m in re.finditer(r"\bFROM\s+([a-zA-Z0-9_]+)", s, re.IGNORECASE):
            used_tables.add(m.group(1).strip().lower())

        for m in re.finditer(r"\bJOIN\s+([a-zA-Z0-9_]+)", s, re.IGNORECASE):
            used_tables.add(m.group(1).strip().lower())

    if not used_tables:
        used_tables = set(schema_map.keys())

    join_tables = set()
    for t1, joins in c.join_keys_by_table.items():
        join_tables.add(t1)
        for (_, t2, _) in joins:
            join_tables.add(t2)

    required_eq_by_table: Dict[str, Dict[str, Any]] = {}
    for (tbl, col), val in c.required_equals.items():
        required_eq_by_table.setdefault(tbl, {})[col] = val

    join_edges = set()
    for t1, joins in c.join_keys_by_table.items():
        for (c1, t2, c2) in joins:
            join_edges.add((t1, c1, t2, c2))

    # NEW — build join components
    from collections import defaultdict

    join_components = defaultdict(set)

    for t1, joins in c.join_keys_by_table.items():
        for (c1, t2, c2) in joins:
            join_components[t1].add((t1, c1))
            join_components[t1].add((t2, c2))

    forced_join_values = {}

    for comp in join_components.values():
        shared = canonical_join_value("JOIN", id(comp))
        for (tbl, col) in comp:
            forced_join_values[(tbl, col)] = shared

    for (tbl, col), val in c.required_equals.items():
        if tbl not in schema_map:
            continue

        for (t1, c1, t2, c2) in join_edges:
            if tbl == t1:
                forced_join_values[(t1, c1)] = forced_join_values.get((t1, c1), f"REQJOIN_{c1}")
                forced_join_values[(t2, c2)] = forced_join_values[(t1, c1)]
            elif tbl == t2:
                forced_join_values[(t2, c2)] = forced_join_values.get((t2, c2), f"REQJOIN_{c2}")
                forced_join_values[(t1, c1)] = forced_join_values[(t2, c2)]

    for (tbl, col), val in c.required_equals.items():

        forced_join_values[(tbl, col)] = val

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

    numeric_sql_cols = infer_numeric_columns_from_sql(sql)
    numeric_sql_cols = {c.lower() for c in numeric_sql_cols}
    normalized_schema: Dict[str, Dict[str, str]] = {}

    for tbl, cols in schema_map.items():
        tbl_lower = tbl.lower()
        new_cols: Dict[str, str] = {}

        for col, raw_dtype in cols.items():
            col_lower = col.lower() 
            raw = (raw_dtype or "").lower()

            if col.lower() in numeric_sql_cols:
                base = "int"

            else:
                base = None

                if any(tok in raw for tok in ("int", "integer")):
                    base = "int"
                elif any(tok in raw for tok in ("real", "float", "double", "numeric", "decimal")):
                    base = "float"
                elif "date" in raw or "time" in raw:
                    base = "date"

            val = required_eq_by_table.get(tbl_lower, {}).get(col)
            if val is not None:
                if isinstance(val, int):
                    base = "int"
                elif isinstance(val, float):
                    base = "float"
                elif isinstance(val, str):
                    if re.match(r"\d{4}(-\d{2}-\d{2})?", val):
                        base = "int" if len(val) == 4 else "date"
                    else:
                        base = "text"

            if base is None:
                base = _infer_dtype(col)

            new_cols[col_lower] = base


        normalized_schema[tbl_lower] = new_cols

    schema_map = normalized_schema

    dataset: Dict[str, pd.DataFrame] = {}

    for table_name, columns in schema_map.items():

        if (
            table_name.lower() not in used_tables
            and table_name.lower() not in join_tables
        ):
            if columns:
                dataset[table_name] = pd.DataFrame([{col: None for col in columns}])
            else:
                dataset[table_name] = pd.DataFrame([{"placeholder": None}])
            continue

        table_required_eq = required_eq_by_table.get(table_name, {})
        
        rows: List[Dict[str, Any]] = []

        #Scenario (test case) handling
        def scenario_rows(scn_name: str) -> List[Dict[str, Any]]:
            """Return rows for special edge-case scenarios."""
            out = []

            def det(col, dtype, idx):
                return _coerce_value_for_dtype(_deterministic_value(col, dtype, row_idx=idx), dtype)

            def rnd(col, dtype):
                t = dtype.lower()
                if "int" in t:
                    return random.randint(1, 100)
                if "float" in t or "real" in t:
                    return random.random() * 10
                if "date" in t:
                    return f"19{random.randint(70,99)}-{random.randint(1,12):02d}-{random.randint(1,28):02d}"
                return _random_string("v")

            def join_val(col):
                key = (table_name, col)
                return global_join_values.get(key, None)

            if scn_name == "no_match":
                for idx in range(n_rows_per_table):
                    r = {}
                    for col, dtype in columns.items():
                        r[col] = det(col, dtype, idx) if is_deterministic else rnd(col, dtype)
                    for (t1, c1, t2, c2) in join_edges:
                        if t1 == table_name:
                            r[c1] = _random_string("noj")
                        if t2 == table_name:
                            r[c2] = _random_string("noj")
                    out.append(r)
                return out

            if scn_name == "single_match":
                r = {}
                for col, dtype in columns.items():
                    key = (table_name, col)

                    if key in forced_join_values or key in global_join_values:
                        r[col] = _coerce_value_for_dtype(
                            forced_join_values.get(key, global_join_values.get(key)),
                            dtype
                        )
                        continue

                    val = get_required_value(table_name, col, dtype, witnesses, 0)
                    if val is not None:
                        r[col] = val
                        continue

                    like_val = get_like_match(table_name, col, c)
                    if like_val is not None:
                        r[col] = _coerce_value_for_dtype(like_val, dtype)
                        continue

                    for (tbl, rc_col, op, val) in c.range_constraints:
                        if tbl.lower() == table_name.lower() and rc_col.lower() == col.lower():
                            if dtype == "date":
                                base = pd.to_datetime(val)
                                r[col] = (base + pd.Timedelta(days=1)).date()

                            elif is_numeric_dtype(dtype):
                                num = coerce_numeric(val, None)
                                if op == "BETWEEN" and isinstance(val, (tuple, list)):
                                    lo = coerce_numeric(val[0], None)
                                    hi = coerce_numeric(val[1], None)
                                    if lo is not None and hi is not None and lo < hi:
                                        r[col] = random.randint(lo + 1, hi - 1)
                                    else:
                                        r[col] = random.randint(10, 100)
                                elif num is not None:
                                    r[col] = num + 1
                                else:
                                    r[col] = random.randint(10, 100)

                            else:
                                r[col] = val
                            break

                    like_val = get_like_match(table_name, col, c)
                    if like_val is not None:
                        r[col] = _coerce_value_for_dtype(like_val, dtype)
                        continue
                        
                    else:
                        r[col] = det(col, dtype, 0)


            if scn_name == "multi_match":

                count = max(3, int(n_rows_per_table * 0.3))
                shared_key = _random_string("mj")

                # --- matching rows ---
                for idx in range(count):
                    r = {}
                    for col, dtype in columns.items():
                        key = (table_name, col)

                        # 1) join keys: same shared key across tables
                        if key in global_join_values or key in forced_join_values:
                            r[col] = _coerce_value_for_dtype(shared_key, dtype)
                            continue

                        is_witness_row = (idx == WITNESS_ROW_INDEX[table_name])

                        if is_witness_row:
                            for (tbl, rc_col, op, val) in c.range_constraints:
                                if tbl == table_name and rc_col == col:
                                    if op == "=":
                                        r[col] = _coerce_value_for_dtype(val, dtype)
                                    num = coerce_numeric(val, None)
                                    if num is not None:
                                        if op == ">":
                                            r[col] = _coerce_value_for_dtype(num + 1, dtype)
                                        elif op == "<":
                                            r[col] = _coerce_value_for_dtype(num - 1, dtype)
                                    elif op == ">=":
                                        r[col] = _coerce_value_for_dtype(val, dtype)
                                    elif op == "<=":
                                        r[col] = _coerce_value_for_dtype(val, dtype)
                                    break

                        val = get_required_value(table_name, col, dtype, witnesses, 0)
                        if val is not None:
                            r[col] = val
                            continue

                        applied = False
                        for (tbl, rc_col, op, val) in c.range_constraints:
                            if tbl.lower() == table_name.lower() and rc_col.lower() == col.lower():
                                if dtype in ("int", "float"):

                                    if op == "BETWEEN":
                                        lo = coerce_numeric(val[0], None)
                                        hi = coerce_numeric(val[1], None)
                                        if lo is not None and hi is not None and lo < hi:
                                            r[col] = random.randint(lo + 1, hi - 1)
                                        else:
                                            r[col] = random.randint(10, 100)

                                    else:
                                        num = coerce_numeric(val, None)
                                        if num is not None:
                                            if op == ">":
                                                r[col] = num + 1
                                            elif op == ">=":
                                                r[col] = num
                                            elif op == "<":
                                                r[col] = num - 1
                                            elif op == "<=":
                                                r[col] = num
                                        else:
                                            r[col] = random.randint(10, 100)

                                applied = True
                                break

                        if applied:
                            continue

                        like_val = get_like_match(table_name, col, c)
                        if like_val is not None:
                            r[col] = _coerce_value_for_dtype(like_val, dtype)
                            continue

                        for (tbl, rc_col, op, val) in c.range_constraints:
                            if tbl == table_name and rc_col == col:
                                if op == "=":
                                    r[col] = _coerce_value_for_dtype(val, dtype)
                                elif op == ">":
                                    r[col] = _coerce_value_for_dtype(val + 1, dtype)
                                elif op == "<":
                                    r[col] = _coerce_value_for_dtype(val - 1, dtype)
                                break

                        r[col] = det(col, dtype, idx) if is_deterministic else rnd(col, dtype)

                    out.append(r)

                for idx in range(count, n_rows_per_table):
                    r = {}
                    for col, dtype in columns.items():
                        key = (table_name, col)

                        if key in global_join_values or key in forced_join_values:
                            r[col] = _coerce_value_for_dtype(shared_key, dtype)
                            continue

                        pushed = False
                        for (tbl, rc_col, op, val) in c.range_constraints:
                            if (
                                tbl.lower() == table_name.lower()
                                and rc_col.lower() == col.lower()
                                and dtype in ("int", "float")
                            ):
                                if op == "BETWEEN":
                                    lo, hi = val
                                    lo_i = int(lo)
                                    hi_i = int(hi)
                                    if random.random() < 0.5:
                                        r[col] = random.randint(1, lo_i - 1) if lo_i > 1 else hi_i + random.randint(1, 100)
                                    else:
                                        r[col] = hi_i + random.randint(1, 100)
                                elif op == ">":
                                    r[col] = val
                                elif op == ">=":
                                    if isinstance(val, (int, float)):
                                        num = coerce_numeric(val, None)
                                        if num is not None:
                                            r[col] = num - 1
                                        else:
                                            r[col] = random.randint(1, 100)
                                    else:
                                        r[col] = random.randint(1, 100)
                                elif op == "<":
                                    if isinstance(val, (int, float)):
                                        num = coerce_numeric(val, None)
                                        if num is not None:
                                            r[col] = num + 1
                                        else:
                                            r[col] = random.randint(1, 100)
                                    else:
                                        r[col] = random.randint(1, 100)
                                elif op == "<=":
                                    if isinstance(val, (int, float)):
                                        num = coerce_numeric(val, None)
                                        if num is not None:
                                            r[col] = num + 1
                                        else:
                                            r[col] = random.randint(1, 100)
                                    else:
                                        r[col] = random.randint(1, 100)
                                pushed = True
                                break

                        if pushed:
                            continue

                        r[col] = det(col, dtype, idx) if is_deterministic else rnd(col, dtype)

                    out.append(r)

                return out

            if scn_name == "duplicate_join_keys":
                dup_val = _random_string("dup")

                for idx in range(n_rows_per_table):
                    r = {}
                    for col, dtype in columns.items():
                        key = (table_name, col)
                        if key in global_join_values or key in forced_join_values:
                            r[col] = _coerce_value_for_dtype(dup_val, dtype)
                        else:
                            r[col] = det(col, dtype, idx) if is_deterministic else rnd(col, dtype)
                    out.append(r)
                return out

            if scn_name == "join_key_nulls":
                for idx in range(n_rows_per_table):
                    r = {}
                    for col, dtype in columns.items():
                        key = (table_name, col)
                        if key in global_join_values or key in forced_join_values:
                            if idx % 2 == 0:
                                r[col] = None
                            else:
                                r[col] = _coerce_value_for_dtype(
                                    forced_join_values.get(key, global_join_values.get(key)),
                                    dtype
                                )
                        else:
                            r[col] = det(col, dtype, idx) if is_deterministic else rnd(col, dtype)
                    out.append(r)
                return out

            if scn_name == "inconsistent_categories":
                for idx in range(n_rows_per_table):
                    r = {}
                    for col, dtype in columns.items():
                        if any(kw in col.lower() for kw in ["type", "status", "category", "option"]):
                            r[col] = random.choice(["A", "B", "C", "UNKNOWN"])
                        else:
                            r[col] = det(col, dtype, idx) if is_deterministic else rnd(col, dtype)
                    out.append(r)
                return out

            if scn_name == "join_only":
                for idx in range(n_rows_per_table):
                    r = {}
                    for col, dtype in columns.items():
                        key = (table_name, col)
                        if key in global_join_values or key in forced_join_values:
                            r[col] = _coerce_value_for_dtype(
                                forced_join_values.get(key, global_join_values.get(key)),
                                dtype
                            )
                        else:
                            r[col] = rnd(col, dtype)
                    out.append(r)
                return out

            return None

        if scenario and scenario.get("name"):
            custom = scenario_rows(scenario["name"])
            if custom is not None:
                df = pd.DataFrame(custom)

                # Scenario inject (mutation)
                if scenario.get("inject"):
                    df = scenario["inject"](df)

                dataset[table_name] = df
                continue

        must_force = (
            bool(table_required_eq)
            or any((table_name, col) in forced_join_values for col in columns)
        )

        row_idx = 0

        if must_force:
            forced_row = {}
            for col, dtype in columns.items():
                key = (table_name, col)

                if (table_name, col) in c.required_equals:
                    forced_row[col] = _coerce_value_for_dtype(
                        c.required_equals[(table_name, col)],
                        dtype
                    )
                    continue

                like_val = get_like_match(table_name, col, c)
                if like_val is not None:
                    forced_row[col] = _coerce_value_for_dtype(like_val, dtype)
                    continue

                if key in forced_join_values:
                    forced_row[col] = _coerce_value_for_dtype(forced_join_values[key], dtype)
                    continue
                if key in global_join_values:
                    forced_row[col] = _coerce_value_for_dtype(global_join_values[key], dtype)
                    continue

                applied = False
                for (tbl, rc_col, op, val) in c.range_constraints:
                    if tbl.lower() == table_name.lower() and rc_col.lower() == col.lower():
                        if dtype in ("int", "float"):
                            if op == "BETWEEN":
                                lo = coerce_numeric(val[0], None)
                                hi = coerce_numeric(val[1], None)

                                if lo is not None and hi is not None and lo < hi:
                                    forced_row[col] = random.randint(lo + 1, hi - 1)
                                else:
                                    forced_row[col] = random.randint(10, 100)

                            else:
                                num = coerce_numeric(val, None)

                                if num is not None:
                                    if op == ">":
                                        forced_row[col] = num + 1
                                    elif op == ">=":
                                        forced_row[col] = num
                                    elif op == "<":
                                        forced_row[col] = num - 1
                                    elif op == "<=":
                                        forced_row[col] = num
                                else:
                                    forced_row[col] = random.randint(10, 100)
                        applied = True
                        break

                if applied:
                    continue

                forced_row[col] = _coerce_value_for_dtype(
                    _deterministic_value(col, dtype, row_idx=row_idx),
                    dtype
                ) if is_deterministic else rnd(col, dtype)

            rows.append(forced_row)
            row_idx += 1

        # matching rows
        match_count = scenario.get("positive_count") if (scenario and "positive_count" in scenario) else max(1, int(n_rows_per_table * match_ratio))

        for _ in range(match_count):
            r = {}
            for col, dtype in columns.items():
                key = (table_name, col)

                val = get_required_value(table_name, col, dtype, witnesses, 0)
                if val is not None:
                    r[col] = val
                    continue

                like_val = get_like_match(table_name, col, c)
                if like_val is not None:
                    r[col] = _coerce_value_for_dtype(like_val, dtype)
                    continue

                if key in global_join_values:
                    r[col] = _coerce_value_for_dtype(global_join_values[key], dtype)
                    continue

                applied = False
                for (tbl, rc_col, op, val) in c.range_constraints:
                    if tbl.lower() == table_name.lower() and rc_col.lower() == col.lower():
                        if dtype in ("int", "float"):
                            if op == "BETWEEN":
                                lo, hi = val
                                r[col] = random.randint(lo + 1, hi - 1)
                            elif op == ">":
                                if isinstance(val, (int, float)):
                                    r[col] = val + 1
                                else:
                                    r[col] = random.randint(1, 100)
                            elif op == ">=":
                                r[col] = val
                            elif op == "<":
                                if isinstance(val, (int, float)):
                                    r[col] = val - 1
                                else:
                                    r[col] = random.randint(1, 100)
                            elif op == "<=":
                                r[col] = val
                        applied = True
                        break

                if applied:
                    continue

                r[col] = _coerce_value_for_dtype(
                    _deterministic_value(col, dtype, row_idx=row_idx),
                    dtype
                ) if is_deterministic else rnd(col, dtype)

            rows.append(r)
            row_idx += 1

        # non-matching rows
        for _ in range(n_rows_per_table - match_count):
            r = {}
            for col, dtype in columns.items():
                key = (table_name, col)
                if key in global_join_values:
                    r[col] = _coerce_value_for_dtype(global_join_values[key], dtype)
                else:
                    r[col] = _coerce_value_for_dtype(
                        _deterministic_value(col, dtype, row_idx=row_idx),
                        dtype
                    ) if is_deterministic else rnd(col, dtype)
            rows.append(r)
            row_idx += 1

        df = pd.DataFrame(rows)

        if scenario and scenario.get("inject"):
            df = scenario["inject"](df)

        dataset[table_name] = df


    for tbl, df in dataset.items():
        for col, dtype in schema_map[tbl].items():
            if dtype == "date":
                def fix_date(x):
                    if x is None:
                        return None
                    try:
                        return pd.to_datetime(x, errors="coerce").date()
                    except Exception:
                        return None

                df[col] = df[col].apply(fix_date)

    _sanitize_division_denominators(dataset, sql)
    _enforce_string_equalities(dataset, constraints)

    for (outer_tbl, outer_col, inner_tbl, inner_col) in c.scalar_subqueries:
        if (
            outer_tbl in dataset and
            inner_tbl in dataset and
            outer_col in dataset[outer_tbl].columns and
            inner_col in dataset[inner_tbl].columns
        ):
            shared_val = f"SCALAR_WINNER"
            dataset[outer_tbl].loc[0, outer_col] = shared_val
            dataset[inner_tbl].loc[0, inner_col] = shared_val

    offset = getattr(c, "offset", None)

    if offset is not None and order_info:
        if order_info and order_info.get("limit"):
            order_col = order_info["expr"].split(".")[-1].replace("`", "")
            direction = order_info["direction"]

            for tbl, df in dataset.items():
                if order_col in df.columns:
                    if direction == "ASC":
                        df.loc[0, order_col] = -1_000_000
                    else:
                        df.loc[0, order_col] = 1_000_000
        order_col = order_info["expr"].split(".")[-1].replace("`", "")
        direction = order_info["direction"]

        PAD = offset

        for tbl, df in dataset.items():
            if order_col not in df.columns:
                continue
            
            padding_rows = []
            for i in range(PAD):
                row = {}
                for col in df.columns:
                    if col == order_col:
                        val = 1_000_000 + i if direction == "DESC" else -1_000_000 - i
                    else:
                        val = df[col].iloc[0]
                    row[col] = val
                padding_rows.append(row)

            dataset[tbl] = pd.concat(
                [pd.DataFrame(padding_rows), df],
                ignore_index=True
            )

    if hasattr(c, "or_groups") and c.or_groups:
        _enforce_or_branch(dataset, schema_map, c.or_groups)

    for t1, joins in c.join_keys_by_table.items():
        if t1 not in dataset:
            continue

        df1 = dataset[t1]
        if len(df1) == 0:
            continue

        for (c1, t2, c2) in joins:
            if t2 not in dataset:
                continue

            df2 = dataset[t2]
            if len(df2) == 0:
                continue

            if c1 not in df1.columns or c2 not in df2.columns:
                continue

            i1 = WITNESS_ROW_INDEX[t1]
            i2 = WITNESS_ROW_INDEX[t2]

            v = df1.loc[i1, c1]
            if pd.isna(v):
                v = forced_join_values.get((t1, c1), f"WINNER_JOIN_{t1}_{c1}")
                df1.loc[i1, c1] = v

            df2.loc[i2, c2] = v

        if getattr(c, "set_op", None) == "INTERSECT":
            for col in c.projected_cols:
                shared_val = f"INTERSECT_{col}"
                for tbl, df in dataset.items():
                    if col in df.columns:
                        df.loc[0, col] = shared_val

            for tbl, df in dataset.items():
                if len(df) == 0:
                    continue
                df.loc[len(df)] = df.loc[0].copy()


    if (
        not scenario
        or scenario.get("name") != "no_match"
    ) and is_query_satisfiable(c):
        enforce_winner_constraints(
            dataset,
            c,
            winner_idx=0,
            allow_joins=(getattr(c, "set_op", None) not in {"EXCEPT", "INTERSECT"}),
            allow_scalar_subqueries=(getattr(c, "set_op", None) != "INTERSECT"),
        )

    for (tbl, col, pat) in getattr(c, "like_constraints", []):
        if tbl in dataset and col in dataset[tbl].columns:
            dataset[tbl].loc[0, col] = f"xxx{pat.strip('%')}yyy"

    if getattr(c, "group_by_cols", None):
        for (tbl, col) in c.group_by_cols:
            if tbl in dataset and col in dataset[tbl].columns:
                dataset[tbl] = pd.concat(
                    [dataset[tbl], dataset[tbl].iloc[[0]].copy()],
                    ignore_index=True
                )

    return dataset