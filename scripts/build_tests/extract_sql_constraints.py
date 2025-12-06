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
            "order_by": (colname, direction) or None,
            "limit": N or None,
            "used_tables": set([table1, table2, ...]),
        }
    """

    # ============================================================
    # Initialize constraint containers
    # ============================================================

    constraints = {
        "required_equals": {},
        "required_not_equals": {},
        "required_is_null": set(),
        "required_not_null": set(),
        "join_keys": set(),
        "range_constraints": [],
        "order_by": None,
        "limit": None,
        "used_tables": set(),
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
        re.IGNORECASE
    )

    for real, alias in alias_pattern.findall(s_clean):
        real = real.split(".")[-1].lower()  # strip schema prefix if any
        if alias:
            alias_map[alias.lower()] = real
        constraints["used_tables"].add(real)

    # Add identity mappings
    for tbl in schema_map:
        alias_map.setdefault(tbl.lower(), tbl.lower())

    # ============================================================
    # STEP 2 — Extract tables from subqueries:
    #   WHERE x = (SELECT ... FROM tbl ...)
    #   SELECT ... FROM (SELECT ... FROM tbl) AS X
    # ============================================================

    subquery_from_pattern = re.compile(
        r"\(\s*SELECT.*?\bFROM\s+([a-zA-Z0-9_\.]+)",
        re.IGNORECASE | re.DOTALL,
    )

    for tbl in subquery_from_pattern.findall(s_clean):
        tbl = tbl.split(".")[-1].lower()
        constraints["used_tables"].add(tbl)

    # ============================================================
    # STEP 3 — Resolve ORDER BY and LIMIT
    # ============================================================

    # ORDER BY col [ASC|DESC]
    order_by_pattern = re.compile(
        r"ORDER\s+BY\s+([a-zA-Z0-9_\.`]+)\s*(ASC|DESC)?",
        re.IGNORECASE
    )
    m = order_by_pattern.search(s_clean)
    if m:
        col = m.group(1).replace("`", "")
        direction = m.group(2).upper() if m.group(2) else "ASC"
        constraints["order_by"] = (col, direction)

    # LIMIT N
    limit_pattern = re.compile(r"LIMIT\s+([0-9]+)", re.IGNORECASE)
    m = limit_pattern.search(s_clean)
    if m:
        constraints["limit"] = int(m.group(1))

    # ============================================================
    # STEP 4 — Extract WHERE clause contents
    # ============================================================

    where_index = s_upper.find(" WHERE ")
    if where_index != -1:
        where_clause = s_clean[where_index + 7:]

        # truncate at ORDER BY or LIMIT
        for kw in [" ORDER BY ", " LIMIT "]:
            idx = where_clause.upper().find(kw)
            if idx != -1:
                where_clause = where_clause[:idx]

        predicates = [p.strip() for p in where_clause.split("AND")]
    else:
        predicates = []

    # ============================================================
    # STEP 5 — Helper: resolve unqualified columns using schema_map
    # ============================================================

    def resolve_unqualified(col):
        col_l = col.lower()
        matches = []
        for t, cols in schema_map.items():
            for c in cols:
                if c.lower() == col_l:
                    matches.append((t, c))
        return matches

    # ============================================================
    # STEP 6 — Parse WHERE predicates
    # ============================================================

    pred_pattern = re.compile(
        r'(?P<table>[a-zA-Z0-9_]+)?\.?'
        r'(?P<column>`[^`]+`|[a-zA-Z0-9_]+)\s*'
        r'(?P<op>=|!=|<>|<=|>=|<|>|IS\s+NOT|IS)\s*'
        r'(?P<value>.+)$',
        re.IGNORECASE
    )


    for pred in predicates:
        m = pred_pattern.search(pred)
        if not m:
            continue

        raw_table = m.group("table")
        col = m.group("column").strip("`")
        op = m.group("op").upper()
        val = m.group("value").strip().strip("'").strip('"')

        # Resolve table name
        if raw_table:
            table = alias_map.get(raw_table.lower())
            if table is None:
                continue
        else:
            # Unqualified column → must match exactly 1 table
            matches = resolve_unqualified(col)
            if len(matches) != 1:
                continue  # ambiguous or not found
            table, col = matches[0]

        if table not in schema_map:
            continue

        key = (table, col)

        if op == "=":
            constraints["required_equals"][key] = val
        elif op in ("!=", "<>"):
            constraints["required_not_equals"][key] = val
        elif op in (">", "<", ">=", "<="):
            constraints["range_constraints"].append(
            (table, col, op, val)
            )
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
        r'([a-zA-Z0-9_]+)\.([`a-zA-Z0-9_]+)\s*=\s*([a-zA-Z0-9_]+)\.([`a-zA-Z0-9_]+)',
        re.IGNORECASE
    )

    for t1, c1, t2, c2 in join_pattern.findall(s_clean):
        real1 = alias_map.get(t1.lower())
        real2 = alias_map.get(t2.lower())
        if not real1 or not real2:
            continue

        c1 = c1.strip("`")
        c2 = c2.strip("`")

        if real1 in schema_map and real2 in schema_map:
            constraints["join_keys"].add((real1, c1, real2, c2))
    print("[parse_constraints][ranges]", constraints["range_constraints"])

    return constraints
