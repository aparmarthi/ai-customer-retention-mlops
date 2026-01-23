from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

from config import PATHS, SETTINGS, ensure_dirs


def _assert_exists(p: Path) -> None:
    if not p.exists():
        raise FileNotFoundError(f"Missing file: {p}")


def main() -> None:
    print("Python executable:", sys.executable)
    print("Python version:", sys.version)
    print("-" * 80)

    ensure_dirs()

    _assert_exists(PATHS.TRAIN_CSV)
    _assert_exists(PATHS.MEMBERS_PARQUET)

    # ----------------------------
    # Load train (small; pandas OK)
    # ----------------------------
    print(f"Loading train: {PATHS.TRAIN_CSV}")
    train = pd.read_csv(PATHS.TRAIN_CSV)

    # Basic checks
    required_train_cols = {"msno", "is_churn"}
    missing = required_train_cols - set(train.columns)
    if missing:
        raise ValueError(f"train.csv missing required columns: {missing}")

    if train["msno"].duplicated().any():
        print("⚠️ train.csv has duplicate msno rows. (Not fatal for now, but unexpected.)")

    # ----------------------------
    # Load members (parquet)
    # ----------------------------
    print(f"Loading members parquet: {PATHS.MEMBERS_PARQUET}")
    members = pd.read_parquet(PATHS.MEMBERS_PARQUET)

    if "msno" not in members.columns:
        raise ValueError("members_v3 missing 'msno' column")

    # ----------------------------
    # Minimal cleaning (safe now)
    # ----------------------------
    # Gender: fill missing with 'unknown'
    if "gender" in members.columns:
        members["gender"] = members["gender"].fillna("unknown").astype(str)

    # Age (bd): invalidate unrealistic ages -> NA
    if "bd" in members.columns:
        bd = members["bd"]
        # If bd is non-numeric for some reason, coerce
        bd = pd.to_numeric(bd, errors="coerce")
        bd_invalid = (bd < SETTINGS.MIN_AGE) | (bd > SETTINGS.MAX_AGE)
        members["bd"] = bd.mask(bd_invalid)

    # registration_init_time: optionally parse to datetime (kept as datetime)
    # Many KKBox files store this as YYYYMMDD integer.
    if "registration_init_time" in members.columns:
        reg = pd.to_numeric(members["registration_init_time"], errors="coerce")
        # Convert YYYYMMDD -> datetime; keep NaT for invalids
        members["registration_init_time_dt"] = pd.to_datetime(reg, format="%Y%m%d", errors="coerce")

    # ----------------------------
    # Build spine = train LEFT JOIN members
    # ----------------------------
    print("Building spine (train LEFT JOIN members on msno)...")
    spine = train.merge(members, on="msno", how="left", validate="m:1")

    print("Spine shape:", spine.shape)
    print("Spine churn rate:", spine["is_churn"].mean())

    # Save
    out = PATHS.SPINE_PARQUET
    spine.to_parquet(out, index=False)
    print(f"✅ Wrote spine parquet: {out}")


if __name__ == "__main__":
    main()
