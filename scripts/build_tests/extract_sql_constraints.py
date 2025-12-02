import sqlparse
import re
from collections import defaultdict


def parse_constraints(sql: str, schema_map: dict):
    """
    Extract equality constraints, inequality constraints, NULL tests,
    and JOIN keys, with correct handling of table aliases.

    Output format:
        {
            "required_equals": { (table, col) : value },
            "required_not_equals": { (table, col) : value },
            "required_is_null": set( (table, col) ),
            "required_not_null": set( (table, col) ),
            "join_keys": set( (t1, c1, t2, c2) )
        }
    """

    constraints = {
        "required_equals": {},
        "required_not_equals": {},
        "required_is_null": set(),
        "required_not_null": set(),
        "join_keys": set(),
    }

    # ----------------------------------------------------------------------
    # STEP 1 — Extract alias map (T1 → frpm, T2 → schools)
    # ----------------------------------------------------------------------
    alias_map = {}

    alias_regex = re.compile(
        r'(FROM|JOIN)\s+([a-zA-Z0-9_]+)\s+(?:AS\s+)?([a-zA-Z0-9_]+)',
        re.IGNORECASE
    )

    for m in alias_regex.finditer(sql):
        _, real_table, alias = m.groups()
        alias_map[alias.lower()] = real_table.lower()

    # Also map real table names to themselves
    for tbl in schema_map:
        alias_map.setdefault(tbl.lower(), tbl.lower())

    # ----------------------------------------------------------------------
    # STEP 2 — Locate WHERE clause
    # ----------------------------------------------------------------------
    sql_upper = sql.upper()
    where_index = sql_upper.find(" WHERE ")
    if where_index == -1:
        # SQL with no WHERE clause still may have JOINs
        pass
    else:
        # Full WHERE → end before ORDER BY / LIMIT if present
        where_section = sql[where_index + 7:]

        for kw in [" ORDER BY ", " LIMIT "]:
            idx = where_section.upper().find(kw)
            if idx != -1:
                where_section = where_section[:idx]

        predicates = [p.strip() for p in where_section.split("AND")]
    # If no WHERE found:
    if where_index == -1:
        predicates = []

    # ----------------------------------------------------------------------
    # STEP 3 — Regex for equality/inequality/NULL predicates
    # ----------------------------------------------------------------------
    pred_pattern = re.compile(
        r'(?P<table>[a-zA-Z0-9_]+)?\.?'
        r'(?P<column>`[^`]+`|[a-zA-Z0-9_]+)\s*'
        r'(?P<op>=|!=|<>|IS|IS NOT)\s*'
        r'(?P<value>.+)$'
    )

    # Resolve table for an unqualified column
    def resolve_unqualified(col):
        col_l = col.lower()
        matches = []
        for t, cols in schema_map.items():
            for c in cols:
                if c.lower() == col_l:
                    matches.append((t, c))
        return matches

    # ----------------------------------------------------------------------
    # STEP 4 — Process each WHERE predicate
    # ----------------------------------------------------------------------
    for pred in predicates:
        m = pred_pattern.search(pred)
        if not m:
            continue

        table = m.group("table")
        col = m.group("column").strip("`")
        op = m.group("op").upper()
        raw = m.group("value").strip()

        # Clean literal value
        val = raw.strip().strip("'").strip('"')

        # Alias → real table
        if table:
            table = alias_map.get(table.lower(), table.lower())
        else:
            # No table specified → resolve
            matches = resolve_unqualified(col)
            if len(matches) == 1:
                table, col = matches[0]
            else:
                continue  # ambiguous or not found

        if table not in schema_map:
            continue

        key = (table, col)

        if op == "=":
            constraints["required_equals"][key] = val
        elif op in ("!=", "<>"):
            constraints["required_not_equals"][key] = val
        elif op == "IS":
            if val.upper() == "NULL":
                constraints["required_is_null"].add(key)
        elif op == "IS NOT":
            if val.upper() == "NULL":
                constraints["required_not_null"].add(key)

    # ----------------------------------------------------------------------
    # STEP 5 — Extract JOIN KEYS (with alias resolution)
    # ----------------------------------------------------------------------
    join_pattern = re.compile(
        r'([a-zA-Z0-9_]+)\.`?([^`]+)`?\s*=\s*([a-zA-Z0-9_]+)\.`?([^`]+)`?',
        re.IGNORECASE
    )

    for m in join_pattern.finditer(sql):
        t1, c1, t2, c2 = m.groups()

        t1 = alias_map.get(t1.lower(), t1.lower())
        t2 = alias_map.get(t2.lower(), t2.lower())

        if t1 in schema_map and t2 in schema_map:
            constraints["join_keys"].add((t1, c1, t2, c2))

    return constraints
