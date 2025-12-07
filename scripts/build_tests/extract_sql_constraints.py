# scripts/build_tests/parse_constraints.py

import re
from collections import defaultdict


def parse_constraints(sql: str, schema_map: dict):
    """
    Extract constraints from SQL, resolving aliases, JOIN keys,
    WHERE predicates, ORDER BY, LIMIT, and used tables (including subqueries).

    Returns dictionary:
        {
            "required_equals": { (table, col): value },
            "required_not_equals": { (table, col): value },
            "required_is_null": set((table, col)),
            "required_not_null": set((table, col)),
            "join_keys": set((t1, c1, t2, c2)),
            "required_join_keys": set((t1, c1, t2, c2)),
            "range_constraints": [(table, col, op, val), ...],
            "order_by": (colname, direction) or None,
            "limit": N or None,
            "used_tables": set([table1, table2, ...]),
            "scalar_subqueries": set((outer_tbl, outer_col, inner_tbl, inner_col)),
        }
    """

    constraints = {
        "required_equals": {},
        "required_not_equals": {},
        "required_is_null": set(),
        "required_not_null": set(),
        "join_keys": set(),
        "required_join_keys": set(),
        "range_constraints": [],
        "order_by": None,
        "limit": None,
        "used_tables": set(),
        "scalar_subqueries": set(),
    }

    # Normalize SQL
    s_clean = sql.replace("\n", " ")
    s_upper = s_clean.upper()

    # ============================================================
    # STEP 1 — Extract aliases (FROM table AS X, JOIN table Y, etc.)
    # ============================================================

    alias_map = {}

    alias_pattern = re.compile(
        r"\b(?:FROM|JOIN)\s+([a-zA-Z0-9_\.]+)"
        r"(?:\s+(?:AS\s+)?([a-zA-Z0-9_]+))?",
        re.IGNORECASE,
    )

    for real, alias in alias_pattern.findall(s_clean):
        real_tbl = real.split(".")[-1]
        real_l = real_tbl.lower()
        if alias:
            alias_map[alias.lower()] = real_l
        constraints["used_tables"].add(real_l)

    # Add identity mappings from schema_map
    for tbl in schema_map:
        alias_map.setdefault(tbl.lower(), tbl.lower())

    # ============================================================
    # STEP 2 — Extract tables from subqueries
    # ============================================================

    subquery_from_pattern = re.compile(
        r"\(\s*SELECT.*?\bFROM\s+([a-zA-Z0-9_\.]+)",
        re.IGNORECASE | re.DOTALL,
    )

    for tbl in subquery_from_pattern.findall(s_clean):
        tbl = tbl.split(".")[-1].lower()
        constraints["used_tables"].add(tbl)

    # ============================================================
    # STEP 3 — ORDER BY and LIMIT
    # ============================================================

    order_by_pattern = re.compile(
        r"ORDER\s+BY\s+([a-zA-Z0-9_\.`]+)\s*(ASC|DESC)?",
        re.IGNORECASE,
    )
    m = order_by_pattern.search(s_clean)
    if m:
        col = m.group(1).replace("`", "")
        direction = m.group(2).upper() if m.group(2) else "ASC"
        constraints["order_by"] = (col, direction)

    limit_pattern = re.compile(r"LIMIT\s+([0-9]+)", re.IGNORECASE)
    m = limit_pattern.search(s_clean)
    if m:
        constraints["limit"] = int(m.group(1))

    # ============================================================
    # STEP 4 — WHERE clause
    # ============================================================

    where_index = s_upper.find(" WHERE ")
    if where_index != -1:
        where_clause = s_clean[where_index + 7 :]
        for kw in [" ORDER BY ", " LIMIT ", " GROUP BY "]:
            idx = where_clause.upper().find(kw)
            if idx != -1:
                where_clause = where_clause[:idx]
        predicates = [p.strip() for p in where_clause.split("AND")]
    else:
        predicates = []

    constraints["has_or"] = False
    constraints["or_groups"] = []

    if where_index != -1:
        constraints["has_or"] = " OR " in where_clause.upper()

        if constraints["has_or"]:
            or_groups = []
            or_clauses = re.split(r"\s+OR\s+", where_clause, flags=re.IGNORECASE)
            for oc in or_clauses:
                preds = [p.strip() for p in oc.split("AND")]
                or_groups.append(preds)
            constraints["or_groups"] = or_groups



    # ============================================================
    # STEP 5 — helper: resolve unqualified columns
    # ============================================================

    def resolve_unqualified(col):
        col_l = col.lower()
        matches = []
        for t, cols in schema_map.items():
            for c in cols:
                if c.lower() == col_l:
                    matches.append((t.lower(), c))
        return matches

    # ============================================================
    # STEP 4.5 — scalar subquery equality detection
    # ============================================================

    scalar_subq_pattern = re.compile(
        r'(?:(?P<t1>[a-zA-Z0-9_]+)\.)?(?P<c1>[a-zA-Z0-9_]+)\s*=\s*\(\s*SELECT\s+'
        r'(?P<c2>[a-zA-Z0-9_]+)\s+FROM\s+(?P<t2>[a-zA-Z0-9_]+)',
        re.IGNORECASE,
    )

    for m in scalar_subq_pattern.finditer(s_clean):
        t1_raw = m.group("t1")
        c1 = m.group("c1")
        c2 = m.group("c2")
        t2_raw = m.group("t2")

        t2 = alias_map.get(t2_raw.lower(), t2_raw.lower())

        if t1_raw:
            outer_tbl = alias_map.get(t1_raw.lower(), t1_raw.lower())
            outer_col = c1
        else:
            matches = resolve_unqualified(c1)
            if len(matches) != 1:
                continue
            outer_tbl, outer_col = matches[0]

        inner_tbl = t2
        inner_col = c2

        outer_tbl = outer_tbl.lower()
        inner_tbl = inner_tbl.lower()

        if outer_tbl in {t.lower() for t in schema_map} and inner_tbl in {
            t.lower() for t in schema_map
        }:
            constraints["scalar_subqueries"].add(
                (outer_tbl, outer_col, inner_tbl, inner_col)
            )
            constraints["join_keys"].add(
                (outer_tbl, outer_col, inner_tbl, inner_col)
            )
            constraints["required_join_keys"].add(
                (outer_tbl, outer_col, inner_tbl, inner_col)
            )

    # ============================================================
    # STEP 6 — WHERE predicates (including LIKE)
    # ============================================================

    pred_pattern = re.compile(
        r"(?P<table>[a-zA-Z0-9_]+)?\.?"
        r"(?P<column>`[^`]+`|[a-zA-Z0-9_]+)\s*"
        r"(?P<op>=|!=|<>|<=|>=|<|>|LIKE|IS\s+NOT|IS)\s*"
        r"(?P<value>.+)$",
        re.IGNORECASE,
    )

    for pred in predicates:
        m = pred_pattern.search(pred)
        if not m:
            continue

        raw_table = m.group("table")
        col = m.group("column").strip("`")
        op = m.group("op").upper()
        val = m.group("value").strip().strip("'").strip('"')

        if raw_table:
            table = alias_map.get(raw_table.lower())
            if table is None:
                continue
        else:
            matches = resolve_unqualified(col)
            if len(matches) != 1:
                continue
            table, col = matches[0]

        table = table.lower()
        if table not in {t.lower() for t in schema_map}:
            continue

        key = (table, col)

        if op in ("=", "LIKE"):
            # treat LIKE 'Riverside%' as a required literal that satisfies it
            if val.isdigit():
                val_conv = int(val)
            else:
                try:
                    val_conv = float(val)
                except Exception:
                    val_conv = val
            constraints["required_equals"][key] = val_conv

        elif op in ("!=", "<>"):
            constraints["required_not_equals"][key] = val

        elif op in (">", "<", ">=", "<="):
            constraints["range_constraints"].append((table, col, op, val))

        elif op == "IS":
            if val.upper() == "NULL":
                constraints["required_is_null"].add(key)

        elif op == "IS NOT":
            if val.upper() == "NULL":
                constraints["required_not_null"].add(key)

    # ============================================================
    # STEP 7 — Extract JOIN keys
    # ============================================================

    join_pattern = re.compile(
        r"([a-zA-Z0-9_]+)\.([`a-zA-Z0-9_]+)\s*=\s*([a-zA-Z0-9_]+)\.([`a-zA-Z0-9_]+)",
        re.IGNORECASE,
    )

    for t1, c1, t2, c2 in join_pattern.findall(s_clean):
        real1 = alias_map.get(t1.lower())
        real2 = alias_map.get(t2.lower())
        if not real1 or not real2:
            continue

        c1 = c1.strip("`")
        c2 = c2.strip("`")

        real1 = real1.lower()
        real2 = real2.lower()

        if real1 in {t.lower() for t in schema_map} and real2 in {
            t.lower() for t in schema_map
        }:
            constraints["join_keys"].add((real1, c1, real2, c2))


    offset_pattern = re.compile(r"LIMIT\s+(\d+)\s*,\s*(\d+)", re.IGNORECASE)
    m = offset_pattern.search(s_clean)
    if m:
        constraints["offset"] = int(m.group(1))
        constraints["offset_limit"] = int(m.group(2))
    else:
        constraints["offset"] = None

    return constraints
