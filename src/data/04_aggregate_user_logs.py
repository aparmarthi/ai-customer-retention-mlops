"""
Step 4: Aggregate User Logs (per-user features)

Purpose:
- Read user logs (user_logs.parquet preferred; fallback to CSV)
- Aggregate to 1 row per msno
- Write: data/kkbox/processed/log_features.parquet

This version is optimized for:
- Large user_logs (e.g., ~8.6GB parquet)
- Windows stability (avoids common mmap/IO issues)
- Speed via 2-stage aggregation: (msno, date) -> (msno)

How to run (from project root, with venv activated):
    python src/data/04_aggregate_user_logs.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import duckdb

from config import PATHS, SETTINGS, ensure_dirs


def _assert_exists(p: Path) -> None:
    if not p.exists():
        raise FileNotFoundError(f"Missing file: {p}")


def _duckdb_connect() -> duckdb.DuckDBPyConnection:
    """
    Use a file-backed DuckDB database so spills work reliably on Windows.
    Configure temp dir + memory budget. Try disabling mmap (helps on Windows for huge Parquets).
    """
    PATHS.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    db_path = PATHS.PROCESSED_DIR / "kkbox.duckdb"
    con = duckdb.connect(database=str(db_path))

    # Temp directory: prefer user setting; else use a short path under processed/
    if SETTINGS.DUCKDB_TEMP_DIR:
        temp_dir = Path(SETTINGS.DUCKDB_TEMP_DIR)
    else:
        temp_dir = PATHS.PROCESSED_DIR / "duckdb_tmp"
    temp_dir.mkdir(parents=True, exist_ok=True)

    con.execute(f"PRAGMA temp_directory='{temp_dir.as_posix()}';")

    # With 64GB RAM, 32GB is a good balance (leave headroom for OS + cache).
    con.execute("PRAGMA memory_limit='32GB';")

    # Reduce memory pressure during hashing/aggregation
    con.execute("PRAGMA preserve_insertion_order=false;")

    # Windows workaround: disabling mmap often avoids "Insufficient system resources" on large files.
    # Not all DuckDB builds support this pragma; ignore if unsupported.
    try:
        con.execute("PRAGMA disable_mmap=true;")
    except Exception:
        pass

    return con


def _available_cols_from_source(con: duckdb.DuckDBPyConnection, from_sql: str) -> set[str]:
    """
    Return column names from a FROM expression like read_parquet('...') without materializing data.
    """
    rows = con.execute(f"DESCRIBE SELECT * FROM {from_sql} LIMIT 0;").fetchall()
    return {r[0] for r in rows}


def main() -> None:
    print("Python executable:", sys.executable)
    print("Python version:", sys.version)
    print("-" * 80)

    ensure_dirs()

    # Prefer parquet if available; otherwise use CSV directly
    if PATHS.USER_LOGS_PARQUET.exists():
        source_path = PATHS.USER_LOGS_PARQUET
        source_type = "parquet"
    else:
        source_path = PATHS.USER_LOGS_CSV
        source_type = "csv"

    _assert_exists(source_path)

    # ✅ Correct output for Step 04
    out_path = PATHS.LOG_FEATURES_PARQUET
    out_path.parent.mkdir(parents=True, exist_ok=True)

    con = _duckdb_connect()
    try:
        print(f"Loading user logs from {source_type}: {source_path}")

        src = source_path.as_posix()
        if source_type == "parquet":
            ul_from = f"read_parquet('{src}')"
        else:
            ul_from = f"""
                read_csv_auto('{src}',
                              header=true,
                              ignore_errors=true,
                              sample_size=200000)
            """.strip()

        cols = _available_cols_from_source(con, ul_from)

        # Required for our 2-stage approach:
        required = {"msno", "date"}
        missing = required - cols
        if missing:
            raise ValueError(
                f"user_logs is missing required columns for this optimized aggregator: {missing}. "
                "If your schema differs, paste the column list and I’ll adjust the script."
            )

        # Columns we will use if available (typical KKBox):
        needed_metrics = {"total_secs", "num_unq", "num_25", "num_50", "num_75", "num_985", "num_100"}
        missing_metrics = sorted(list(needed_metrics - cols))
        if missing_metrics:
            raise ValueError(
                "Your user_logs schema is missing one or more expected metric columns used by this aggregator: "
                f"{missing_metrics}. Paste your column list and I’ll generate a schema-adaptive version."
            )

        print("Aggregating user logs using 2-stage strategy: (msno, date) -> (msno) ...")

        # 2-stage aggregation dramatically reduces the intermediate data size and speeds up final GROUP BY.
        agg_sql = f"""
        COPY (
            WITH daily AS (
                SELECT
                    msno,
                    date,
                    COUNT(*) AS row_cnt_d,
                    SUM(CAST(total_secs AS DOUBLE)) AS total_secs_sum_d,
                    SUM(CAST(num_unq AS DOUBLE)) AS num_unq_sum_d,
                    SUM(CAST(num_25 AS DOUBLE)) AS num_25_sum_d,
                    SUM(CAST(num_50 AS DOUBLE)) AS num_50_sum_d,
                    SUM(CAST(num_75 AS DOUBLE)) AS num_75_sum_d,
                    SUM(CAST(num_985 AS DOUBLE)) AS num_985_sum_d,
                    SUM(CAST(num_100 AS DOUBLE)) AS num_100_sum_d
                FROM {ul_from}
                GROUP BY msno, date
            )
            SELECT
                msno,

                -- counts / dates
                SUM(row_cnt_d) AS log_row_cnt,
                MIN(date) AS log_first_date,
                MAX(date) AS log_last_date,
                COUNT(*) AS log_active_days,
                MAX(date) - MIN(date) AS log_span_days_approx,

                -- seconds: totals and per-active-day stats
                SUM(total_secs_sum_d) AS total_secs_sum,
                AVG(total_secs_sum_d) AS total_secs_mean_per_active_day,
                MAX(total_secs_sum_d) AS total_secs_max_per_day,

                -- unique tracks: totals and per-active-day stats
                SUM(num_unq_sum_d) AS num_unq_sum,
                AVG(num_unq_sum_d) AS num_unq_mean_per_active_day,
                MAX(num_unq_sum_d) AS num_unq_max_per_day,

                -- bucket totals
                SUM(num_25_sum_d) AS num_25_sum,
                SUM(num_50_sum_d) AS num_50_sum,
                SUM(num_75_sum_d) AS num_75_sum,
                SUM(num_985_sum_d) AS num_985_sum,
                SUM(num_100_sum_d) AS num_100_sum,

                -- ratios
                CASE
                    WHEN (SUM(num_25_sum_d) + SUM(num_50_sum_d) + SUM(num_75_sum_d) + SUM(num_985_sum_d) + SUM(num_100_sum_d)) > 0
                    THEN SUM(num_100_sum_d) * 1.0
                         / (SUM(num_25_sum_d) + SUM(num_50_sum_d) + SUM(num_75_sum_d) + SUM(num_985_sum_d) + SUM(num_100_sum_d))
                    ELSE NULL
                END AS listen_full_share,

                CASE
                    WHEN SUM(num_unq_sum_d) > 0
                    THEN (SUM(num_25_sum_d) + SUM(num_50_sum_d) + SUM(num_75_sum_d) + SUM(num_985_sum_d) + SUM(num_100_sum_d)) * 1.0
                         / SUM(num_unq_sum_d)
                    ELSE NULL
                END AS listens_per_unique_track

            FROM daily
            GROUP BY msno
        )
        TO '{out_path.as_posix()}'
        (FORMAT PARQUET, CODEC 'ZSTD');
        """

        con.execute(agg_sql)

        # Quick sanity checks
        n_rows = con.execute(
            f"SELECT COUNT(*) FROM read_parquet('{out_path.as_posix()}');"
        ).fetchone()[0]

        print(f"✅ Wrote log features: {out_path}")
        print(f"Rows in log_features (unique users with logs): {n_rows:,}")

        preview = con.execute(
            f"SELECT * FROM read_parquet('{out_path.as_posix()}') LIMIT 3;"
        ).fetchdf()
        print("\nPreview (first 3 rows):")
        print(preview)

    finally:
        con.close()


if __name__ == "__main__":
    main()
