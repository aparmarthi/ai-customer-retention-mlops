from __future__ import annotations

import sys
from pathlib import Path

import duckdb

from config import PATHS, SETTINGS, ensure_dirs


def _assert_exists(p: Path) -> None:
    if not p.exists():
        raise FileNotFoundError(f"Missing file: {p}")


def convert_csv_to_parquet(csv_path: Path, parquet_path: Path) -> None:
    """
    Convert CSV -> Parquet using DuckDB without loading full file into pandas.
    """
    print(f"Converting:\n  CSV: {csv_path}\n  -> PARQUET: {parquet_path}")

    con = duckdb.connect(database=":memory:")
    try:
        if SETTINGS.DUCKDB_TEMP_DIR:
            con.execute(f"PRAGMA temp_directory='{SETTINGS.DUCKDB_TEMP_DIR}';")

        # read_csv_auto infers schema; SAMPLE_SIZE=-1 can be slow on huge files.
        # DuckDB default is usually fine; if inference issues, increase sample_size.
        con.execute(
            f"""
            COPY (
                SELECT *
                FROM read_csv_auto('{csv_path.as_posix()}',
                                  header=true,
                                  ignore_errors=true,
                                  sample_size=200000)
            )
            TO '{parquet_path.as_posix()}'
            (FORMAT PARQUET);
            """
        )
    finally:
        con.close()

    print("✅ Done.\n")


def main() -> None:
    print("Python executable:", sys.executable)
    print("Python version:", sys.version)
    print("-" * 80)

    ensure_dirs()

    # Validate raw files exist
    _assert_exists(PATHS.MEMBERS_CSV)
    _assert_exists(PATHS.TRANSACTIONS_CSV)
    _assert_exists(PATHS.USER_LOGS_CSV)

    # Convert
    convert_csv_to_parquet(PATHS.MEMBERS_CSV, PATHS.MEMBERS_PARQUET)
    convert_csv_to_parquet(PATHS.TRANSACTIONS_CSV, PATHS.TRANSACTIONS_PARQUET)
    convert_csv_to_parquet(PATHS.USER_LOGS_CSV, PATHS.USER_LOGS_PARQUET)

    print("All parquet conversions complete.")
    print(f"Parquet directory: {PATHS.PARQUET_DIR}")


if __name__ == "__main__":
    main()
