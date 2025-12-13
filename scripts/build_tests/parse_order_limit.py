import re

#HANDLES ORDER AND LIMIT QUERIES SPECIFICALLY

def parse_order_limit(sql: str):

    s = " ".join(sql.replace("\n", " ").split())

    limit_match = re.search(r"LIMIT\s+(\d+)", s, re.IGNORECASE)
    if not limit_match:
        return None

    limit_val = int(limit_match.group(1))

    order_match = re.search(
        r"ORDER BY\s+(.+?)(?:\s+(ASC|DESC))?\s+LIMIT",
        s,
        re.IGNORECASE
    )

    if not order_match:
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
