from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, List

import duckdb
import pandas as pd
import datetime

#UNIT TEST STRUCTURE

def _json_safe(o):
    if isinstance(o, (datetime.date, datetime.datetime)):
        return o.isoformat()

    try:
        import numpy as np
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, (np.bool_)):
            return bool(o)
    except:
        pass

    return str(o)


@dataclass
class SQLUnitTest:
    db_id: str
    query_index: int
    scenario: str
    sql: str
    tables: Dict[str, pd.DataFrame]
    expected_output: pd.DataFrame

    def to_json(self, path: Path) -> None:
        payload = {
            "db_id": self.db_id,
            "query_index": self.query_index,
            "scenario": self.scenario,
            "sql": self.sql,
            "tables": {
                t: df.to_dict(orient="records") for t, df in self.tables.items()
            },
            "expected_output": self.expected_output.to_dict(orient="records"),
        }

        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            import json

            json.dump(payload, f, indent=2, default=_json_safe)

    @staticmethod
    def from_json(path: Path) -> "SQLUnitTest":
        import json

        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)

        tables = {
            t: pd.DataFrame(rows) for t, rows in payload["tables"].items()
        }
        expected = pd.DataFrame(payload["expected_output"])

        return SQLUnitTest(
            db_id=payload["db_id"],
            query_index=payload["query_index"],
            scenario=payload["scenario"],
            sql=payload["sql"],
            tables=tables,
            expected_output=expected,
        )

    def _run_sql(self, sql_query: str) -> pd.DataFrame | None:
        con = duckdb.connect()
        try:
            for t, df in self.tables.items():
                con.register(t, df)

            duckdb_query = sql_query.replace("`", '"')
            result = con.execute(duckdb_query).df()
            return result
        except Exception as e:
            print(f"[SQLUnitTest] SQL execution error: {e}")
            return None
        finally:
            con.close()

    def compute_expected(self) -> pd.DataFrame | None:
        return self._run_sql(self.sql)

    def check(self, candidate_sql: str) -> bool:
        gold = self.compute_expected()
        cand = self._run_sql(candidate_sql)

        if gold is None or cand is None:
            return False

        def normalize(df: pd.DataFrame) -> pd.DataFrame:
            df2 = df.copy()
            df2 = df2.sort_index(axis=1)
            if df2.shape[0] > 0:
                df2 = df2.sort_values(
                    by=list(df2.columns), ignore_index=True
                )
            return df2

        gold_norm = normalize(gold)
        cand_norm = normalize(cand)
        return gold_norm.equals(cand_norm)
