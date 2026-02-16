from __future__ import annotations
from unicodedata import name
import sys
from pathlib import Path

import duckdb
import pandas as pd

from config import PATHS, SETTINGS, ensure_dirs

"""
Create sample CSVs for derived tables:
- spine.parquet (from step 02)
- model_table.parquet (from step 05)

This script samples by selecting N msnos from the existing model_table (or spine)
and then subsetting both tables to the same msnos.

Outputs (CSV) under: data/kkbox/sample/
- spine_sample.csv
- model_table_sample.csv
- README_derived_samples.md

Run (from repo root, venv activated):
    python src/data/07_create_derived_samples.py
"""

N_USERS = 1000
RANDOM_SEED = 42


def _assert_exists(p: Path) -> None:
    if not p.exists():
        raise FileNotFoundError(f"Missing file: {p}")


def _duckdb_connect(sample_dir: Path) -> duckdb.DuckDBPyConnection:
    db_path = sample_dir / "derived_samples.duckdb"
    con = duckdb.connect(database=str(db_path))

    tmp_dir = sample_dir / "duckdb_tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    if SETTINGS.DUCKDB_TEMP_DIR:
        try:
            con.execute(f"PRAGMA temp_directory='{Path(SETTINGS.DUCKDB_TEMP_DIR).as_posix()}';")
        except Exception:
            con.execute(f"PRAGMA temp_directory='{tmp_dir.as_posix()}';")
    else:
        con.execute(f"PRAGMA temp_directory='{tmp_dir.as_posix()}';")

    con.execute("PRAGMA memory_limit='4GB';")

    try:
        con.execute("PRAGMA threads=16;")
    except Exception:
        pass

    try:
        con.execute("PRAGMA disable_mmap=true;")
    except Exception:
        pass

    return con


def _copy_query_to_csv(con: duckdb.DuckDBPyConnection, query_sql: str, out_csv: Path) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    con.execute(
        f"""
        COPY (
            {query_sql}
        )
        TO '{out_csv.as_posix()}'
        (HEADER, DELIMITER ',');
        """
    )


def main() -> None:
    print("Python executable:", sys.executable)
    print("Python version:", sys.version)
    print("-" * 80)

    ensure_dirs()

    sample_dir = PATHS.KKBOX_DIR / "sample"
    sample_dir.mkdir(parents=True, exist_ok=True)

    spine_path = PATHS.SPINE_PARQUET
    model_table_path = PATHS.PROCESSED_DIR / "model_table.parquet"

    _assert_exists(spine_path)
    _assert_exists(model_table_path)

    con = _duckdb_connect(sample_dir)
    try:
        # Choose msnos from model_table for consistency
        print(f"Sampling {N_USERS} users from model_table: {model_table_path}")
        msno_df = con.execute(
            f"""
            SELECT msno
            FROM read_parquet('{model_table_path.as_posix()}')
            USING SAMPLE reservoir({N_USERS})
            """
        ).fetchdf()

        if msno_df.empty:
            raise ValueError("Could not sample msnos from model_table.parquet (table might be empty).")

        # Ensure unique msnos
        msno_df = msno_df.dropna().drop_duplicates()
        if len(msno_df) == 0:
            raise ValueError("Sampled msnos are empty after dropna/drop_duplicates.")

        msno_path = sample_dir / "msno_derived_sample.parquet"
        msno_df.to_parquet(msno_path, index=False)

        con.execute(f"CREATE VIEW msno_list AS SELECT * FROM read_parquet('{msno_path.as_posix()}');")

        # Write spine sample
        spine_out = sample_dir / "spine_sample.csv"
        print(f"Writing spine_sample.csv from: {spine_path}")
        _copy_query_to_csv(
            con,
            f"""
            SELECT sp.*
            FROM read_parquet('{spine_path.as_posix()}') sp
            INNER JOIN msno_list s USING(msno)
            """.strip(),
            spine_out,
        )
        print(f"✅ Wrote: {spine_out}")

        # Write model table sample
        model_out = sample_dir / "model_table_sample.csv"
        print(f"Writing model_table_sample.csv from: {model_table_path}")
        _copy_query_to_csv(
            con,
            f"""
            SELECT mt.*
            FROM read_parquet('{model_table_path.as_posix()}') mt
            INNER JOIN msno_list s USING(msno)
            """.strip(),
            model_out,
        )
        print(f"✅ Wrote: {model_out}")

    finally:
        con.close()

    readme_path = sample_dir / "README_derived_samples.md"
    readme_path.write_text(
        f"""# Derived Samples (Spine + Model Table)

These files are **small samples** of the pipeline-derived tables.

## Purpose
- Provide a quick look at the final engineered feature tables
- Allow reviewers to inspect the modeling schema without downloading full datasets
- Not intended for training real models

## Sampling strategy
- Sampled **{N_USERS}** users (`msno`) from `model_table.parquet`
- Subset both `spine.parquet` and `model_table.parquet` to that same `msno` list

## Files
- spine_sample.csv (<= {N_USERS} users; 1 row per msno)
- model_table_sample.csv (<= {N_USERS} users; 1 row per msno)

## Generate
Run from project root (with venv activated):
```bash
python src/data/07_create_derived_samples.py
""",
encoding="utf-8",
)
print(f"✅ Wrote: {readme_path}")
print("Done.")

if __name__ == "__main__":
    main()