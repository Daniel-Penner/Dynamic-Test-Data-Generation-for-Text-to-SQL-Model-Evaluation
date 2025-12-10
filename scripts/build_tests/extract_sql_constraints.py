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

    set_ops = [
        s.upper()
        for s in re.findall(r"\b(INTERSECT|UNION|EXCEPT)\b", sql)
    ]
    global_set_op = set_ops[0] if set_ops else None

# ============================================================
# PRE-STEP — split on set operators (INTERSECT / UNION / EXCEPT)
# ============================================================

    set_split = re.split(
        r"\b(INTERSECT|UNION|EXCEPT)\b",
        sql,
        flags=re.IGNORECASE,
    )

    select_blocks = [
        blk.strip()
        for blk in set_split
        if blk.strip().upper().startswith("SELECT")
    ]

    all_constraints = []

    for sql_part in select_blocks:

        constraints_part = {
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

        constraints_part["set_op"] = global_set_op
        constraints_part["projected_cols"] = []

        s_clean = sql_part.replace("\n", " ")
        s_upper = s_clean.upper()

        m = re.search(r"SELECT\s+(.*?)\s+FROM", s_clean, re.IGNORECASE)
        if m:
            cols = []
            for c in m.group(1).split(","):
                cols.append(
                    c.strip()
                    .split(".")[-1]
                    .strip("`")
                )
            constraints_part["projected_cols"] = cols

        # ============================================================
        # STEP 1 — Extract aliases
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
            constraints_part["used_tables"].add(real_l)

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
            constraints_part["used_tables"].add(tbl)

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
            constraints_part["order_by"] = (col, direction)

        limit_pattern = re.compile(r"LIMIT\s+([0-9]+)", re.IGNORECASE)
        m = limit_pattern.search(s_clean)
        if m:
            constraints_part["limit"] = int(m.group(1))

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

        constraints_part["has_or"] = False
        constraints_part["or_groups"] = []

        if where_index != -1:
            constraints_part["has_or"] = " OR " in where_clause.upper()

            if constraints_part["has_or"]:
                or_groups = []
                or_clauses = re.split(r"\s+OR\s+", where_clause, flags=re.IGNORECASE)
                for oc in or_clauses:
                    preds = [p.strip() for p in oc.split("AND")]
                    or_groups.append(preds)
                constraints_part["or_groups"] = or_groups



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
                constraints_part["scalar_subqueries"].add(
                    (outer_tbl, outer_col, inner_tbl, inner_col)
                )
                constraints_part["join_keys"].add(
                    (outer_tbl, outer_col, inner_tbl, inner_col)
                )
                constraints_part["required_join_keys"].add(
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

            if op == "=":
                # normal equality
                if val.isdigit():
                    val_conv = int(val)
                else:
                    try:
                        val_conv = float(val)
                    except Exception:
                        val_conv = val

                constraints_part["required_equals"][key] = val_conv

            elif op == "LIKE":
                # store LIKE constraints separately (do NOT treat as equality)
                constraints_part.setdefault("like_constraints", []).append(
                    (table, col, val)
                )

            elif op in ("!=", "<>"):
                constraints_part["required_not_equals"][key] = val

            elif op in (">", "<", ">=", "<="):
                try:
                    val = int(val)
                except:
                    try:
                        val = float(val)
                    except:
                        pass
                constraints_part["range_constraints"].append((table, col, op, val))

            elif op == "IS":
                if val.upper() == "NULL":
                    constraints_part["required_is_null"].add(key)

            elif op == "IS NOT":
                if val.upper() == "NULL":
                    constraints_part["required_not_null"].add(key)

        # ============================================================
        # STEP 6.5 — BETWEEN range constraints
        # ============================================================

        between_pattern = re.compile(
            r"""
            (?:(?P<table>[a-zA-Z0-9_]+)\.)?
            (?P<column>`[^`]+`|[a-zA-Z0-9_]+)
            \s+BETWEEN\s+
            (?P<lo>-?\d+)
            \s+AND\s+
            (?P<hi>-?\d+)
            """,
            re.IGNORECASE | re.VERBOSE,
        )

        for m in between_pattern.finditer(where_clause if where_index != -1 else ""):
            raw_table = m.group("table")
            col = m.group("column").strip("`")
            lo = int(m.group("lo"))
            hi = int(m.group("hi"))

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

            constraints_part["range_constraints"].append(
                (table, col, "BETWEEN", (lo, hi))
            )


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

            c1_raw = c1.strip("`")
            c2_raw = c2.strip("`")

            real1 = real1.lower()
            real2 = real2.lower()

            # ✅ map SQL column → real schema column (case-safe)
            def resolve_col(table, col):
                for real_col in schema_map.get(table, {}):
                    if real_col.lower() == col.lower():
                        return real_col
                return col  # fallback

            c1_real = resolve_col(real1, c1_raw)
            c2_real = resolve_col(real2, c2_raw)

            if real1 in {t.lower() for t in schema_map} and real2 in {
                t.lower() for t in schema_map
            }:
                constraints_part["join_keys"].add(
                    (real1, c1_real, real2, c2_real)
                )


        offset_pattern = re.compile(r"LIMIT\s+(\d+)\s*,\s*(\d+)", re.IGNORECASE)
        m = offset_pattern.search(s_clean)
        if m:
            constraints_part["offset"] = int(m.group(1))
            constraints_part["offset_limit"] = int(m.group(2))
        else:
            constraints_part["offset"] = None

        all_constraints.append(constraints_part)

    # ============================================================
    # FINAL — merge constraints from set operators conservatively
    # ============================================================

    constraints = all_constraints[0]
    constraints.setdefault("like_constraints", [])

    for other in all_constraints[1:]:
        constraints["join_keys"] |= other["join_keys"]
        constraints["required_join_keys"] |= other["required_join_keys"]
        constraints["used_tables"] |= other["used_tables"]

        if constraints.get("set_op") == "INTERSECT":
            constraints["required_equals"].clear()
            constraints["required_not_equals"].clear()
            constraints["required_is_null"].clear()
            constraints["required_not_null"].clear()
            constraints["range_constraints"] = []

        if constraints.get("set_op") != "INTERSECT":
            constraints["range_constraints"].extend(other["range_constraints"])

        constraints["scalar_subqueries"] |= other["scalar_subqueries"]

        # ✅ MERGE LIKE CONSTRAINTS
        if "like_constraints" in other:
            constraints.setdefault("like_constraints", [])
            constraints["like_constraints"].extend(other["like_constraints"])

        if constraints.get("set_op") != "INTERSECT":
            for k, v in list(constraints["required_equals"].items()):
                if k not in other["required_equals"] or other["required_equals"][k] != v:
                    del constraints["required_equals"][k]

        # ✅ FINAL FIX FOR INTERSECT: shared projected columns
        if constraints.get("set_op") == "INTERSECT":
            common_cols = set(all_constraints[0].get("projected_cols", []))

            for c in all_constraints[1:]:
                common_cols &= set(c.get("projected_cols", []))

            shared_val = "__INTERSECT_SHARED__"

            for col in common_cols:
                matches = []
                for t, cols in schema_map.items():
                    if col in cols:
                        matches.append(t.lower())

                if len(matches) == 1:
                    constraints["required_equals"][(matches[0], col)] = shared_val

    return constraints
