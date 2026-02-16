"""
Create small sample CSVs for GitHub (schema + relationship demo)

Strategy:
1) Pick N users (msno) from train
2) Subset members/transactions/user_logs to only those msnos
3) Write small CSVs into: data/kkbox/sample/

Run (from repo root, venv activated):
    python src/data/06_create_sample_dataset.py

Notes:
- Uses DuckDB to subset big tables without loading everything into pandas.
- Output files are intended for schema/demo only (not for real training).
"""

from __future__ import annotations

import sys
from pathlib import Path

import duckdb
import pandas as pd

from config import PATHS, SETTINGS, ensure_dirs


# ---- TUNE THESE ----
N_USERS = 1000  # how many msnos to sample from train
MAX_TXN_ROWS = 5000  # cap overall rows in transactions_sample.csv
MAX_LOG_ROWS = 20000  # cap overall rows in user_logs_sample.csv
RANDOM_SEED = 42
# -------------------


def _assert_exists(p: Path) -> None:
    if not p.exists():
        raise FileNotFoundError(f"Missing file: {p}")


def _duckdb_connect(sample_dir: Path) -> duckdb.DuckDBPyConnection:
    # file-backed DB helps stability on Windows for large scans
    db_path = sample_dir / "sample_build.duckdb"
    con = duckdb.connect(database=str(db_path))

    # temp directory for spills
    tmp_dir = sample_dir / "duckdb_tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    con.execute(f"PRAGMA temp_directory='{tmp_dir.as_posix()}';")

    # reasonable defaults; bump if you want (you have 64GB RAM)
    con.execute("PRAGMA memory_limit='8GB';")
    try:
        con.execute("PRAGMA threads=16;")
    except Exception:
        pass

    # Windows workaround for large parquet scans (ignore if unsupported)
    try:
        con.execute("PRAGMA disable_mmap=true;")
    except Exception:
        pass

    return con


def _choose_source(parquet_path: Path, csv_path: Path) -> tuple[str, Path]:
    """Prefer parquet if exists; otherwise CSV."""
    if parquet_path.exists():
        return "parquet", parquet_path
    return "csv", csv_path


def main() -> None:
    print("Python executable:", sys.executable)
    print("Python version:", sys.version)
    print("-" * 80)

    ensure_dirs()

    # Output folder for samples
    sample_dir = PATHS.KKBOX_DIR / "sample"
    sample_dir.mkdir(parents=True, exist_ok=True)

    # --- 1) Sample msnos from train (train is usually manageable in pandas) ---
    _assert_exists(PATHS.TRAIN_CSV)
    train = pd.read_csv(PATHS.TRAIN_CSV)

    if "msno" not in train.columns or "is_churn" not in train.columns:
        raise ValueError("train.csv must contain msno and is_churn")

    # Stratified sample (keeps churn ratio similar)
    train_sample = (
        train.groupby("is_churn", group_keys=False)
        .apply(lambda g: g.sample(n=max(1, int(N_USERS * len(g) / len(train))), random_state=RANDOM_SEED))
    )

    # If rounding overshoots, trim deterministically
    if len(train_sample) > N_USERS:
        train_sample = train_sample.sample(n=N_USERS, random_state=RANDOM_SEED)

    msnos = train_sample["msno"].dropna().astype(str).unique().tolist()
    print(f"Selected msnos: {len(msnos):,}")

    # Write train_sample
    train_out = sample_dir / "train_sample.csv"
    train_sample.to_csv(train_out, index=False)
    print(f"✅ Wrote: {train_out}")

    # Create a tiny msno list file for DuckDB to join against
    msno_df = pd.DataFrame({"msno": msnos})
    msno_path = sample_dir / "msno_sample.parquet"
    msno_df.to_parquet(msno_path, index=False)

    # --- 2) Subset other tables using DuckDB ---
    con = _duckdb_connect(sample_dir)
    try:
        con.execute(f"CREATE VIEW msno_list AS SELECT * FROM read_parquet('{msno_path.as_posix()}');")

        # MEMBERS
        mem_type, mem_src = _choose_source(PATHS.MEMBERS_PARQUET, PATHS.MEMBERS_CSV)
        _assert_exists(mem_src)
        members_out = sample_dir / "members_sample.csv"
        print(f"\nSubsetting members from {mem_type}: {mem_src}")

        mem_from = (
            f"read_parquet('{mem_src.as_posix()}')"
            if mem_type == "parquet"
            else f"read_csv_auto('{mem_src.as_posix()}', header=true, ignore_errors=true, sample_size=200000)"
        )

        con.execute(f"""
            COPY (
                SELECT m.*
                FROM {mem_from} m
                INNER JOIN msno_list s USING(msno)
            )
            TO '{members_out.as_posix()}'
            (HEADER, DELIMITER ',');
        """)
        print(f"✅ Wrote: {members_out}")

        # TRANSACTIONS
        txn_type, txn_src = _choose_source(PATHS.TRANSACTIONS_PARQUET, PATHS.TRANSACTIONS_CSV)
        _assert_exists(txn_src)
        transactions_out = sample_dir / "transactions_sample.csv"
        print(f"\nSubsetting transactions from {txn_type}: {txn_src}")

        txn_from = (
            f"read_parquet('{txn_src.as_posix()}')"
            if txn_type == "parquet"
            else f"read_csv_auto('{txn_src.as_posix()}', header=true, ignore_errors=true, sample_size=200000)"
        )

        # Limit overall rows to keep Git-friendly size
        con.execute(f"""
            COPY (
                SELECT t.*
                FROM {txn_from} t
                INNER JOIN msno_list s USING(msno)
                LIMIT {MAX_TXN_ROWS}
            )
            TO '{transactions_out.as_posix()}'
            (HEADER, DELIMITER ',');
        """)
        print(f"✅ Wrote: {transactions_out}")

        # USER LOGS
        logs_type, logs_src = _choose_source(PATHS.USER_LOGS_PARQUET, PATHS.USER_LOGS_CSV)
        _assert_exists(logs_src)
        user_logs_out = sample_dir / "user_logs_sample.csv"
        print(f"\nSubsetting user_logs from {logs_type}: {logs_src}")

        logs_from = (
            f"read_parquet('{logs_src.as_posix()}')"
            if logs_type == "parquet"
            else f"read_csv_auto('{logs_src.as_posix()}', header=true, ignore_errors=true, sample_size=200000)"
        )

        # Limit overall rows to keep Git-friendly size
        con.execute(f"""
            COPY (
                SELECT ul.*
                FROM {logs_from} ul
                INNER JOIN msno_list s USING(msno)
                LIMIT {MAX_LOG_ROWS}
            )
            TO '{user_logs_out.as_posix()}'
            (HEADER, DELIMITER ',');
        """)
        print(f"✅ Wrote: {user_logs_out}")

    finally:
        con.close()

    # Optional: write a README for the sample folder
    readme_path = sample_dir / "README.md"
    readme_path.write_text(
        f"""# KKBox Sample Dataset

These are **small sample CSVs** generated from the original KKBox churn dataset.

Purpose:
- Provide a quick view of schema and table relationships
- Allow fast end-to-end pipeline testing on a laptop
- Not intended for training real models

Sampling strategy:
- Selected **{N_USERS}** users (`msno`) from `train.csv` (stratified by `is_churn`)
- Subset `members_v3`, `transactions`, and `user_logs` to those users
- Capped row counts for large tables to keep the repo lightweight:
  - transactions_sample.csv <= {MAX_TXN_ROWS} rows
  - user_logs_sample.csv <= {MAX_LOG_ROWS} rows

Generated by:
- `python src/data/06_create_sample_dataset.py`
""",
        encoding="utf-8",
    )
    print(f"\n✅ Wrote: {readme_path}")

    print("\nDone. You can commit data/kkbox/sample/ to GitHub (small + useful).")


if __name__ == "__main__":
    main()
