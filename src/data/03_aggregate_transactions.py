from __future__ import annotations

import sys
from pathlib import Path

import duckdb

from config import PATHS, SETTINGS, ensure_dirs


def _assert_exists(p: Path) -> None:
    if not p.exists():
        raise FileNotFoundError(f"Missing file: {p}")


def _duckdb_connect() -> duckdb.DuckDBPyConnection:
    con = duckdb.connect(database=":memory:")
    if SETTINGS.DUCKDB_TEMP_DIR:
        con.execute(f"PRAGMA temp_directory='{SETTINGS.DUCKDB_TEMP_DIR}';")
    return con


def _available_cols(con: duckdb.DuckDBPyConnection, table: str) -> set[str]:
    rows = con.execute(f"PRAGMA table_info('{table}');").fetchall()
    # PRAGMA table_info returns: (cid, name, type, notnull, dflt_value, pk)
    return {r[1] for r in rows}


def main() -> None:
    print("Python executable:", sys.executable)
    print("Python version:", sys.version)
    print("-" * 80)

    ensure_dirs()

    # Prefer parquet if available; otherwise use CSV directly
    if PATHS.TRANSACTIONS_PARQUET.exists():
        source_path = PATHS.TRANSACTIONS_PARQUET
        source_type = "parquet"
    else:
        source_path = PATHS.TRANSACTIONS_CSV
        source_type = "csv"

    _assert_exists(source_path)

    out_path = PATHS.TXN_FEATURES_PARQUET

    con = _duckdb_connect()
    try:
        print(f"Loading transactions from {source_type}: {source_path}")

        if source_type == "parquet":
            con.execute(
                f"CREATE VIEW tx AS SELECT * FROM read_parquet('{source_path.as_posix()}');"
            )
        else:
            con.execute(
                f"""
                CREATE VIEW tx AS
                SELECT *
                FROM read_csv_auto('{source_path.as_posix()}',
                                  header=true,
                                  ignore_errors=true,
                                  sample_size=200000);
                """
            )

        cols = _available_cols(con, "tx")
        required = {"msno"}
        missing = required - cols
        if missing:
            raise ValueError(f"transactions missing required columns: {missing}")

        # Build dynamic feature SQL depending on which columns exist
        # Common KKBox transaction columns:
        # - payment_method_id, payment_plan_days, plan_list_price, actual_amount_paid,
        #   is_auto_renew, is_cancel, transaction_date, membership_expire_date
        agg_exprs = [
            "COUNT(*) AS txn_cnt",
        ]

        if "is_cancel" in cols:
            agg_exprs.append("SUM(CAST(is_cancel AS BIGINT)) AS cancel_cnt")
            agg_exprs.append("AVG(CAST(is_cancel AS DOUBLE)) AS cancel_rate")
        else:
            agg_exprs.append("NULL::BIGINT AS cancel_cnt")
            agg_exprs.append("NULL::DOUBLE AS cancel_rate")

        if "is_auto_renew" in cols:
            agg_exprs.append("SUM(CAST(is_auto_renew AS BIGINT)) AS auto_renew_cnt")
            agg_exprs.append("AVG(CAST(is_auto_renew AS DOUBLE)) AS auto_renew_rate")
        else:
            agg_exprs.append("NULL::BIGINT AS auto_renew_cnt")
            agg_exprs.append("NULL::DOUBLE AS auto_renew_rate")

        if "plan_list_price" in cols:
            agg_exprs += [
                "AVG(CAST(plan_list_price AS DOUBLE)) AS plan_list_price_mean",
                "MAX(CAST(plan_list_price AS DOUBLE)) AS plan_list_price_max",
                "MIN(CAST(plan_list_price AS DOUBLE)) AS plan_list_price_min",
            ]
        else:
            agg_exprs += [
                "NULL::DOUBLE AS plan_list_price_mean",
                "NULL::DOUBLE AS plan_list_price_max",
                "NULL::DOUBLE AS plan_list_price_min",
            ]

        if "actual_amount_paid" in cols:
            agg_exprs += [
                "AVG(CAST(actual_amount_paid AS DOUBLE)) AS actual_paid_mean",
                "MAX(CAST(actual_amount_paid AS DOUBLE)) AS actual_paid_max",
                "MIN(CAST(actual_amount_paid AS DOUBLE)) AS actual_paid_min",
            ]
        else:
            agg_exprs += [
                "NULL::DOUBLE AS actual_paid_mean",
                "NULL::DOUBLE AS actual_paid_max",
                "NULL::DOUBLE AS actual_paid_min",
            ]

        if "payment_method_id" in cols:
            agg_exprs.append("COUNT(DISTINCT payment_method_id) AS payment_method_nunique")
        else:
            agg_exprs.append("NULL::BIGINT AS payment_method_nunique")

        # Date features
        if "transaction_date" in cols:
            # transaction_date is often YYYYMMDD int
            agg_exprs += [
                "MIN(transaction_date) AS txn_first_date",
                "MAX(transaction_date) AS txn_last_date",
                "MAX(transaction_date) - MIN(transaction_date) AS txn_tenure_days_approx",
            ]
        else:
            agg_exprs += [
                "NULL::BIGINT AS txn_first_date",
                "NULL::BIGINT AS txn_last_date",
                "NULL::BIGINT AS txn_tenure_days_approx",
            ]

        # membership_expire_date can also exist
        if "membership_expire_date" in cols:
            agg_exprs.append("MAX(membership_expire_date) AS membership_expire_date_max")
        else:
            agg_exprs.append("NULL::BIGINT AS membership_expire_date_max")

        agg_sql = f"""
        COPY (
            SELECT
                msno,
                {", ".join(agg_exprs)}
            FROM tx
            GROUP BY msno
        )
        TO '{out_path.as_posix()}'
        (FORMAT PARQUET);
        """

        print("Aggregating transactions to per-user features (msno)...")
        con.execute(agg_sql)

        # Quick sanity checks
        n_rows = con.execute(f"SELECT COUNT(*) FROM read_parquet('{out_path.as_posix()}');").fetchone()[0]
        print(f"✅ Wrote txn features: {out_path}")
        print(f"Rows in txn_features (unique users with txns): {n_rows:,}")

    finally:
        con.close()


if __name__ == "__main__":
    main()
