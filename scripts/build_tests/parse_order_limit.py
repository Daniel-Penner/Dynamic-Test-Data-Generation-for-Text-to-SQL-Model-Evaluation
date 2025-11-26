# scripts/parse_order_limit.py
import re

def parse_order_limit(sql: str):
    """
    Extract ORDER BY expression, direction (ASC/DESC), and LIMIT N
    from an SQL query.

    Returns:
        {
            "expr": "<ORDER BY expression>",
            "direction": "ASC" or "DESC",
            "limit": <int>
        }
    or None if LIMIT is not present.
    """

    # Normalize whitespace
    s = " ".join(sql.replace("\n", " ").split())

    # ---------------------------------------
    # Extract LIMIT N
    # ---------------------------------------
    limit_match = re.search(r"LIMIT\s+(\d+)", s, re.IGNORECASE)
    if not limit_match:
        return None  # no LIMIT → no ORDER/LIMIT shaping needed

    limit_val = int(limit_match.group(1))

    # ---------------------------------------
    # Extract ORDER BY <expr> [ASC|DESC]
    # ---------------------------------------
    # Pattern:
    #   ORDER BY <anything up to whitespace or LIMIT> [ASC|DESC]
    order_match = re.search(
        r"ORDER BY\s+(.+?)(?:\s+(ASC|DESC))?\s+LIMIT",
        s,
        re.IGNORECASE
    )

    if not order_match:
        # LIMIT but no explicit ORDER BY (unusual but possible)
        return {
            "expr": None,
            "direction": "ASC",
            "limit": limit_val
        }

    expr = order_match.group(1).strip()
    direction = order_match.group(2) or "ASC"

    return {
        "expr": expr,
        "direction": direction.upper(),
        "limit": limit_val
    }
