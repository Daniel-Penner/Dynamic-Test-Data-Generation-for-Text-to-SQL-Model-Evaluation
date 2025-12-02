import sqlparse
import re
from collections import defaultdict


def parse_constraints(sql: str, schema_map: dict):
    """
    Extract equality, inequality, null tests, and join keys from SQL.

    FIXED:
    - Unqualified columns ("County Name") now mapped to the correct table automatically.
    - Constraints stored as (table, column) pairs to avoid ambiguity.
    """

    constraints = {
        "required_equals": {},         # (table, col) -> value
        "required_not_equals": {},     # (table, col) -> value
        "required_is_null": set(),     # (table, col)
        "required_not_null": set(),    # (table, col)
        "join_keys": set(),            # ((t1,c1), (t2,c2))
    }

    # --------------------------------------------------------------
    # Extract WHERE clause
    # --------------------------------------------------------------
    sql_upper = sql.upper()
    where_index = sql_upper.find(" WHERE ")

    if where_index == -1:
        return constraints

    where_section = sql[where_index + 7:]

    # Trim ORDER BY / LIMIT
    for keyword in [" ORDER BY ", " LIMIT "]:
        idx = where_section.upper().find(keyword)
        if idx != -1:
            where_section = where_section[:idx]

    predicates = [p.strip() for p in where_section.split("AND")]

    # --------------------------------------------------------------
    # Regex: table.col op value
    # Handles:
    #   col = 'X'
    #   table.col = 'X'
    #   `County Name` = 'Alameda'
    # --------------------------------------------------------------
    pattern = re.compile(
        r'(?P<table>[a-zA-Z0-9_]+)?\.?'
        r'(?P<column>`[^`]+`|[a-zA-Z0-9_]+)\s*'
        r'(?P<op>=|!=|<>|IS|IS NOT)\s*'
        r'(?P<value>.+)$'
    )

    # --------------------------------------------------------------
    # Helper: resolve which table owns an unqualified column
    # --------------------------------------------------------------
    def resolve_table_for_column(col_name):
        col_lower = col_name.lower()
        matches = []
        for table, cols in schema_map.items():
            for c in cols:
                if c.lower() == col_lower:
                    matches.append((table, c))
        return matches

    # --------------------------------------------------------------
    # EXTRACT SIMPLE PREDICATES
    # --------------------------------------------------------------
    for pred in predicates:
        pred = pred.strip()
        match = pattern.search(pred)
        if not match:
            continue

        table = match.group("table")
        col = match.group("column").strip("`")
        op = match.group("op").upper()
        raw_val = match.group("value").strip()

        # clean `'value'` or `"value"`
        val = raw_val.strip().strip("'").strip('"')

        # ----------------------------------------------------------
        # If table is missing, resolve column across schema
        # ----------------------------------------------------------
        if table is None:
            matches = resolve_table_for_column(col)
            if len(matches) == 1:
                table, col = matches[0]   # resolved correctly
            else:
                # Ambiguous — skip
                continue

        # Ensure table exists
        if table not in schema_map:
            continue

        key = (table, col)

        # ----------------------------------------------------------
        # Save constraint
        # ----------------------------------------------------------
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

    # --------------------------------------------------------------
    # Extract JOIN KEYS
    # --------------------------------------------------------------
    join_pattern = re.compile(
        r'([a-zA-Z0-9_]+)\.`?([^`]+)`?\s*=\s*([a-zA-Z0-9_]+)\.`?([^`]+)`?',
        re.IGNORECASE
    )

    # Build alias → real table map (e.g., "t1"→"frpm")
    alias_map = {}

    # Look for FROM / JOIN aliases
    # Handles e.g. "FROM frpm AS T1" or "JOIN schools T2"
    alias_assign = re.compile(
        r'FROM\s+([a-zA-Z0-9_]+)\s+(?:AS\s+)?([a-zA-Z0-9_]+)|'
        r'JOIN\s+([a-zA-Z0-9_]+)\s+(?:AS\s+)?([a-zA-Z0-9_]+)',
        re.IGNORECASE
    )

    for m in alias_assign.finditer(sql):
        t_real_1, alias_1, t_real_2, alias_2 = m.groups()

        if t_real_1 and alias_1:
            alias_map[alias_1.lower()] = t_real_1.lower()

        if t_real_2 and alias_2:
            alias_map[alias_2.lower()] = t_real_2.lower()

    # --------------------------------------------------------------
    # Now parse join conditions
    # --------------------------------------------------------------
    for m in join_pattern.finditer(sql):
        t1, c1, t2, c2 = m.groups()

        # normalize/resolve aliases
        t1_norm = alias_map.get(t1.lower(), t1.lower())
        t2_norm = alias_map.get(t2.lower(), t2.lower())

        constraints["join_keys"].add(
            (t1_norm, c1, t2_norm, c2)
        )

    return constraints
