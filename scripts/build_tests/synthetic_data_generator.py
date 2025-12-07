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

# =====================================================================
# Helper random generators
# =====================================================================

import re

def canonical_join_value(col: str, idx: int = 0) -> str:
    # stable, identical across tables
    return f"JOINVAL_{idx}"


def _infer_dtype(col_name, example_values=None):
    """
    Infer a coarse dtype: 'int', 'float', 'date', or 'text'
    based on the column name (and optionally example values).
    """

    name = col_name.lower()
    # split into tokens: ["low", "grade"], ["free", "meal", "count", "k", "12"], etc.
    tokens = [t for t in re.split(r"[^a-z0-9]+", name) if t]

    # 1) grade-like stuff → integers
    if "grade" in tokens:
        return "int"

    # 2) clearly numeric aggregate tokens
    numeric_tokens = {"count", "num", "number", "total", "enrollment", "sum"}
    if any(t in numeric_tokens for t in tokens):
        return "int"

    # 3) rates / percents → float
    if "%" in col_name or "percent" in tokens or "ratio" in tokens or "rate" in tokens:
        return "float"

    # 4) latitude / longitude → float
    if "latitude" in tokens or "longitude" in tokens:
        return "float"

    # 5) dates
    if "date" in tokens:
        return "date"

    # 6) fallback
    return "text"

def _deterministic_value(col: str, dtype: str, row_idx: int | None):
    """
    Produce deterministic but *varying* values.
    If row_idx is None (edge-case override), generate a random suffix.
    """
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

    # text-like fallback
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
    """
    Enforce constraints on the *winner row only*.

    - WHERE predicates always apply
    - JOIN constraints apply if allow_joins=True
    - Scalar subqueries apply if allow_scalar_subqueries=True
    """

    # --------------------------------------------------
    # 1) WHERE col = value
    # --------------------------------------------------
    for (tbl, col), val in constraints.required_equals.items():
        if tbl in dataset and col in dataset[tbl].columns:
            df = dataset[tbl]
            if winner_idx < len(df):
                df.loc[winner_idx, col] = val

    # --------------------------------------------------
    # 2) WHERE col IS NOT NULL
    # --------------------------------------------------
    for (tbl, col) in constraints.required_not_null:
        if tbl in dataset and col in dataset[tbl].columns:
            df = dataset[tbl]
            if winner_idx < len(df):
                if pd.isna(dataset[tbl].loc[winner_idx, col]):
                    dataset[tbl].loc[winner_idx, col] = 1

    # --------------------------------------------------
    # 3) JOIN constraints (same-table join map)
    # --------------------------------------------------
    if allow_joins:
        for tbl, join_defs in constraints.join_keys_by_table.items():
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

                # Take a non-null value from either side, propagate to both.
                val = df.loc[winner_idx, col]
                if pd.isna(val):
                    val = other_df.loc[winner_idx, other_col]

                if pd.isna(val):
                    # fall back to a deterministic join key
                    val = f"WINNER_JOIN_{tbl}_{col}"

                df.loc[winner_idx, col] = val
                other_df.loc[winner_idx, other_col] = val

    # --------------------------------------------------
    # 4) Scalar subqueries: outer.col = (SELECT inner.col ...)
    # --------------------------------------------------
    if allow_scalar_subqueries and getattr(constraints, "scalar_subqueries", None):
        for (outer_tbl, outer_col, inner_tbl, inner_col) in constraints.scalar_subqueries:
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

            # Prefer any existing non-null value
            if pd.notna(outer_val):
                shared = outer_val
            elif pd.notna(inner_val):
                shared = inner_val
            else:
                shared = f"SCALAR_WINNER_{outer_tbl}_{outer_col}"

            odf.loc[winner_idx, outer_col] = shared
            idf.loc[winner_idx, inner_col] = shared



def _random_string(prefix: str, length: int = 6) -> str:
    suffix = "".join(random.choices(string.digits, k=length))
    return f"{prefix}_{suffix}"


    # Utility: normal fallback numeric/string generator
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

def generate_join_keys(n: int, prefix: str = "jk") -> list[str]:
    """Generate join keys that match across tables but vary by row."""
    return [f"{prefix}_{i:05d}" for i in range(n)]

def _enforce_or_branch(dataset, schema_map, or_groups, alias_map=None):
    """
    Enforce exactly ONE OR branch on winner row (row 0).
    """
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
            # try resolve by schema
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
    WINNER_IDX = 0
    order_winner_idx: Dict[str, int] = {}

    def is_numeric_dtype(dtype: str) -> bool:
        t = dtype.lower()
        return ("int" in t) or ("real" in t) or ("float" in t) or ("double" in t)


    # NEW — detect whether this scenario is deterministic (no mutation fn)
    is_deterministic = not scenario or ("inject" not in scenario or scenario["inject"] is None)

    # NEW — if deterministic, try to preserve REAL values from schema_map
    original_examples = {
        tbl: list(cols.keys()) for tbl, cols in schema_map.items()
    }

    # =============================================================
    # Normalize input: constraints comes as {required_equals,...}
    # or as a Constraints object
    # =============================================================
    if not isinstance(constraints, Constraints):
        # Build per-table constraint layout using **lowercased** table names
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

        # required_equals
        for (tbl, col), val in constraints.get("required_equals", {}).items():
            tbl_l = tbl.lower()
            if tbl_l in per_table:
                per_table[tbl_l]["required_equals"][col] = val

        # required_not_equals
        for (tbl, col), val in constraints.get("required_not_equals", {}).items():
            tbl_l = tbl.lower()
            if tbl_l in per_table:
                per_table[tbl_l]["required_not_equals"][col] = val

        # is null
        for (tbl, col) in constraints.get("required_is_null", set()):
            tbl_l = tbl.lower()
            if tbl_l in per_table:
                per_table[tbl_l]["required_is_null"].add(col)

        # not null
        for (tbl, col) in constraints.get("required_not_null", set()):
            tbl_l = tbl.lower()
            if tbl_l in per_table:
                per_table[tbl_l]["required_not_null"].add(col)

        # join keys: (t1, c1, t2, c2)
        for (t1, c1, t2, c2) in constraints.get("join_keys", set()):
            t1_l = t1.lower()
            t2_l = t2.lower()
            if t1_l in per_table:
                per_table[t1_l]["join_keys"].add((c1, t2_l, c2))
            if t2_l in per_table:
                per_table[t2_l]["join_keys"].add((c2, t1_l, c1))

        # carry over range_constraints + scalar_subqueries
        range_constraints = constraints.get("range_constraints", [])
        scalar_subqueries = constraints.get("scalar_subqueries", set())

        constraints_obj = Constraints(per_table)
        constraints_obj.range_constraints = list(range_constraints)
        constraints_obj.scalar_subqueries = set(scalar_subqueries)

        constraints = constraints_obj

    c = constraints

    # =============================================================
    # NEW — Construct global witness assignments (query-satisfying rows)
    # =============================================================
    WITNESS_COUNT = 3
    witnesses = plan_witness_assignments(c, k=WITNESS_COUNT)
    winner = witnesses[0]

    used_tables: set[str] = set()
    if sql:
        # Normalize whitespace and strip backticks so `frpm` → frpm
        s = " ".join(sql.replace("\n", " ").split())
        s = s.replace("`", "")

        # FROM <table>
        for m in re.finditer(r"\bFROM\s+([a-zA-Z0-9_]+)", s, re.IGNORECASE):
            used_tables.add(m.group(1).strip().lower())

        # JOIN <table>
        for m in re.finditer(r"\bJOIN\s+([a-zA-Z0-9_]+)", s, re.IGNORECASE):
            used_tables.add(m.group(1).strip().lower())

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

    # =============================================================
    # STEP 1: Identify which table has WHERE constraints
    #         and create forced join rows to guarantee matches
    # =============================================================
    forced_join_values: Dict[Tuple[str, str], Any] = {}

    join_counter = 0
    for (t1, c1, t2, c2) in join_edges:
        shared_val = canonical_join_value(c1, join_counter)
        join_counter += 1

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
    # absolute dtype normalization as fallback
    # DO NOT GUESS TYPES — trust the schema provided by BIRD
    # Just normalize to lowercase table names
    normalized_schema: Dict[str, Dict[str, str]] = {}

    for tbl, cols in schema_map.items():
        tbl_lower = tbl.lower()
        new_cols: Dict[str, str] = {}

        for col, raw_dtype in cols.items():
            raw = (raw_dtype or "").lower()


            # First infer base type from schema / column name
            if any(tok in raw for tok in ("int", "integer")):
                base = "int"
            elif any(tok in raw for tok in ("real", "float", "double", "numeric", "decimal")):
                base = "float"
            elif "date" in raw or "time" in raw:
                base = "date"
            else:
                base = _infer_dtype(col)

            # ✅ OVERRIDE using required_equals
            val = required_eq_by_table.get(tbl_lower, {}).get(col)
            if val is not None:
                if isinstance(val, int):
                    base = "int"
                elif isinstance(val, float):
                    base = "float"
                elif isinstance(val, str):
                    # Detect YYYY or YYYY-MM-DD
                    if re.match(r"\d{4}(-\d{2}-\d{2})?", val):
                        base = "int" if len(val) == 4 else "date"
                    else:
                        base = "text"

            new_cols[col] = base

        normalized_schema[tbl_lower] = new_cols

    # Replace schema_map with normalized version
    schema_map = normalized_schema

    dataset: Dict[str, pd.DataFrame] = {}

    for table_name, columns in schema_map.items():

        # Skip unused tables but preserve schema
        if table_name.lower() not in used_tables:
            if columns:
                dataset[table_name] = pd.DataFrame([{col: None for col in columns}])
            else:
                dataset[table_name] = pd.DataFrame([{"placeholder": None}])
            continue

        table_required_eq = required_eq_by_table.get(table_name, {})
        
        rows: List[Dict[str, Any]] = []

        for wi, w in enumerate(witnesses):
            row = {}
            for col, dtype in columns.items():
                key = (table_name, col)

                if key in w:
                    # witness-assigned value (WHERE, GROUP BY, etc.)
                    row[col] = _coerce_value_for_dtype(w[key], dtype)

                elif key in forced_join_values:
                    # 🔑 CRITICAL: enforce matching join/subquery keys
                    row[col] = _coerce_value_for_dtype(forced_join_values[key], dtype)

                else:
                    row[col] = _coerce_value_for_dtype(
                        _deterministic_value(col, dtype, row_idx=None),
                        dtype
                    )

            rows.append(row)
            if order_info and order_info["expr"]:
                order_col = order_info["expr"].split(".")[-1].replace("`", "")
                direction = order_info["direction"]

                if order_col in row:
                    if wi == 0:
                        order_winner_idx[table_name] = wi
                        row[order_col] = 10_000 if direction == "DESC" else -10_000
                    else:
                        row[order_col] = -10_000 if direction == "DESC" else 10_000



        # ============================================================
        # Scenario handler (scenario-overrides normal generation)
        # ============================================================

        def scenario_rows(scn_name: str) -> List[Dict[str, Any]]:
            """Return rows for special edge-case scenarios."""
            out = []

            # Utility: deterministic with row index
            def det(col, dtype, idx):
                return _coerce_value_for_dtype(_deterministic_value(col, dtype, row_idx=idx), dtype)

            # Utility: normal fallback numeric/string
            def rnd(col, dtype):
                t = dtype.lower()
                if "int" in t:
                    return random.randint(1, 100)
                if "float" in t or "real" in t:
                    return random.random() * 10
                if "date" in t:
                    return f"19{random.randint(70,99)}-{random.randint(1,12):02d}-{random.randint(1,28):02d}"
                return _random_string("v")

            # Utility: join key value retrieval
            def join_val(col):
                key = (table_name, col)
                return global_join_values.get(key, None)

            # ----------------------------------------------------------
            # SCENARIO: no_match → produce rows guaranteed to NOT match
            # ----------------------------------------------------------
            if scn_name == "no_match":
                for idx in range(n_rows_per_table):
                    r = {}
                    for col, dtype in columns.items():
                        r[col] = det(col, dtype, idx) if is_deterministic else rnd(col, dtype)
                    # break forced join by injecting unique keys
                    for (t1, c1, t2, c2) in join_edges:
                        if t1 == table_name:
                            r[c1] = _random_string("noj")
                        if t2 == table_name:
                            r[c2] = _random_string("noj")
                    out.append(r)
                return out

            # ----------------------------------------------------------
            # SCENARIO: single_match → exactly 1 matching row
            # ----------------------------------------------------------
            if scn_name == "single_match":
                # create one matching row
                r = {}
                for col, dtype in columns.items():
                    key = (table_name, col)

                    # JOIN KEYS
                    if key in forced_join_values or key in global_join_values:
                        r[col] = _coerce_value_for_dtype(
                            forced_join_values.get(key, global_join_values.get(key)),
                            dtype
                        )
                        continue

                    # REQUIRED EQUALS
                    if col in table_required_eq:
                        r[col] = _coerce_value_for_dtype(table_required_eq[col], dtype)
                        continue

                    # ✅ RANGE CONSTRAINTS (THIS WAS MISSING)
                    for (tbl, c, op, val) in constraints.range_constraints:
                        if tbl == table_name and c == col:
                            if dtype == "date":
                                base = pd.to_datetime(val)
                                r[col] = (base + pd.Timedelta(days=1)).date()
                            elif is_numeric_dtype(dtype):
                                if isinstance(val, str) and "SELECT" in val.upper():
                                    continue
                                else:
                                    r[col] = float(val) + 1
                            else:
                                r[col] = val
                            break
                    else:
                        # fallback
                        r[col] = det(col, dtype, 0)


            # ----------------------------------------------------------
            # SCENARIO: multi_match → create several aligned matches
            # ----------------------------------------------------------
            if scn_name == "multi_match":
                count = max(3, int(n_rows_per_table * 0.3))
                shared_key = _random_string("mj")

                for idx in range(count):
                    r = {}
                    for col, dtype in columns.items():
                        key = (table_name, col)

                        if key in global_join_values or key in forced_join_values:
                            r[col] = _coerce_value_for_dtype(shared_key, dtype)
                        elif col in table_required_eq:
                            r[col] = _coerce_value_for_dtype(table_required_eq[col], dtype)
                        else:
                            r[col] = det(col, dtype, idx) if is_deterministic else rnd(col, dtype)
                    out.append(r)

                # remaining rows non-match
                for idx in range(count, n_rows_per_table):
                    r = {}
                    for col, dtype in columns.items():
                        r[col] = det(col, dtype, idx) if is_deterministic else rnd(col, dtype)
                    out.append(r)

                return out

            # ----------------------------------------------------------
            # SCENARIO: duplicate_join_keys → repeat the SAME join key
            # ----------------------------------------------------------
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

            # ----------------------------------------------------------
            # SCENARIO: join_key_nulls → nullify join keys in some rows
            # ----------------------------------------------------------
            if scn_name == "join_key_nulls":
                for idx in range(n_rows_per_table):
                    r = {}
                    for col, dtype in columns.items():
                        key = (table_name, col)
                        if key in global_join_values or key in forced_join_values:
                            # half null / half valid
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

            # ----------------------------------------------------------
            # SCENARIO: inconsistent_categories → break category cols
            # ----------------------------------------------------------
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

            # ----------------------------------------------------------
            # SCENARIO: join_only → only join keys matter; everything else noise
            # ----------------------------------------------------------
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

        # ============================================================
        # Scenario override (if scenario has structured meaning)
        # ============================================================
        if scenario and scenario.get("name"):
            custom = scenario_rows(scenario["name"])
            if custom is not None:
                df = pd.DataFrame(custom)

                # Scenario inject (mutation)
                if scenario.get("inject"):
                    df = scenario["inject"](df)

                dataset[table_name] = df
                continue

        # ============================================================
        # FALLBACK: ORIGINAL DEFAULT GENERATION LOGIC
        # ============================================================

        # forced row?
        must_force = (
            bool(table_required_eq)
            or any((table_name, col) in forced_join_values for col in columns)
        )

        row_idx = 0

        if must_force:
            forced_row = {}
            for col, dtype in columns.items():
                key = (table_name, col)

                if col in table_required_eq:
                    forced_row[col] = _coerce_value_for_dtype(table_required_eq[col], dtype)
                elif key in forced_join_values:
                    forced_row[col] = _coerce_value_for_dtype(forced_join_values[key], dtype)
                elif key in global_join_values:
                    forced_row[col] = _coerce_value_for_dtype(global_join_values[key], dtype)
                else:
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

                if col in table_required_eq:
                    r[col] = _coerce_value_for_dtype(table_required_eq[col], dtype)
                elif key in global_join_values:
                    r[col] = _coerce_value_for_dtype(global_join_values[key], dtype)
                else:
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

    # =============================================================
    # FINAL PATCH: sanitize all DATE columns to ensure valid values
    # =============================================================

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

    # --------------------------------------------------
    # ✅ FIX: enforce scalar-subquery equality
    # outer.col == inner.col for winner row
    # --------------------------------------------------
    for (outer_tbl, outer_col, inner_tbl, inner_col) in constraints.scalar_subqueries:
        if (
            outer_tbl in dataset and
            inner_tbl in dataset and
            outer_col in dataset[outer_tbl].columns and
            inner_col in dataset[inner_tbl].columns
        ):
            shared_val = f"SCALAR_WINNER"
            dataset[outer_tbl].loc[0, outer_col] = shared_val
            dataset[inner_tbl].loc[0, inner_col] = shared_val

    offset = getattr(constraints, "offset", None)

    if offset is not None and order_info:
        order_col = order_info["expr"].split(".")[-1].replace("`", "")
        direction = order_info["direction"]

        PAD = offset  # number of dummy rows needed

        for tbl, df in dataset.items():
            if order_col not in df.columns:
                continue
            
            # Build padding rows
            padding_rows = []
            for i in range(PAD):
                row = {}
                for col in df.columns:
                    if col == order_col:
                        # Make these beat the winner in ordering
                        val = 1_000_000 + i if direction == "DESC" else -1_000_000 - i
                    else:
                        # junk but type-safe
                        val = df[col].iloc[0]
                    row[col] = val
                padding_rows.append(row)

            dataset[tbl] = pd.concat(
                [pd.DataFrame(padding_rows), df],
                ignore_index=True
            )

    if hasattr(constraints, "or_groups") and constraints.or_groups:
        _enforce_or_branch(dataset, schema_map, constraints.or_groups)

    # =============================================================
    # ✅ FINAL ENFORCEMENT: force at least ONE satisfying row
    # =============================================================
    if not scenario or scenario.get("name") != "no_match":
        enforce_winner_constraints(
            dataset,
            constraints,
            winner_idx=0,
            allow_joins=True,
            allow_scalar_subqueries=True,
            order_winner_idx=order_winner_idx,
        )





    return dataset