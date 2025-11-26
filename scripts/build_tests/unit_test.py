# scripts/build_tests/unit_test.py

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, List

import duckdb
import pandas as pd


@dataclass
class SQLUnitTest:
    """
    Represents one unit test for a given Text-to-SQL query.

    - db_id: which BIRD database this query belongs to
    - query_index: index within dev.json (0-based)
    - scenario: scenario label, e.g. "happy_path", "over_limit", "no_match"
    - sql: the gold SQL query this test is built for
    - tables: {table_name: DataFrame} synthetic data
    - expected_output: DataFrame with gold query result on synthetic data
    """
    db_id: str
    query_index: int
    scenario: str
    sql: str
    tables: Dict[str, pd.DataFrame]
    expected_output: pd.DataFrame

    # ----------------------------
    # Serialization
    # ----------------------------
    def to_json(self, path: Path) -> None:
        """
        Serialize this test to JSON: dataframes become lists of records.
        """
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

            json.dump(payload, f, indent=2)

    @staticmethod
    def from_json(path: Path) -> "SQLUnitTest":
        """
        Load a SQLUnitTest from JSON.
        """
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

    # ----------------------------
    # Execution helpers
    # ----------------------------
    def _run_sql(self, sql_query: str) -> pd.DataFrame | None:
        """
        Run SQL query on this test's synthetic dataset using DuckDB.
        """
        con = duckdb.connect()
        try:
            # Register each table as a view
            for t, df in self.tables.items():
                con.register(t, df)

            # DuckDB uses double-quotes for identifiers; allow backticks by converting.
            duckdb_query = sql_query.replace("`", '"')
            result = con.execute(duckdb_query).df()
            return result
        except Exception as e:
            print(f"[SQLUnitTest] SQL execution error: {e}")
            return None
        finally:
            con.close()

    def compute_expected(self) -> pd.DataFrame | None:
        """
        Compute the gold output on current tables.
        """
        return self._run_sql(self.sql)

    def check(self, candidate_sql: str) -> bool:
        """
        Returns True if candidate SQL produces the same result as the gold SQL
        on this test's synthetic dataset.
        """
        gold = self.compute_expected()
        cand = self._run_sql(candidate_sql)

        if gold is None or cand is None:
            return False

        # Normalize by sorting rows and columns
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
