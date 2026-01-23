from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Paths:
    # Project root (…/ai-customer-retention-mlops/)
    ROOT: Path = Path(__file__).resolve().parents[1]

    # Data dirs
    DATA_DIR: Path = ROOT / "data"
    RAW_DIR: Path = DATA_DIR / "raw"
    PARQUET_DIR: Path = DATA_DIR / "parquet"
    PROCESSED_DIR: Path = DATA_DIR / "processed"

    # Raw inputs (adjust filenames if yours differ)
    TRAIN_CSV: Path = RAW_DIR / "train.csv"
    MEMBERS_CSV: Path = RAW_DIR / "members_v3.csv"
    TRANSACTIONS_CSV: Path = RAW_DIR / "transactions.csv"
    USER_LOGS_CSV: Path = RAW_DIR / "user_logs.csv"

    # Parquet outputs
    MEMBERS_PARQUET: Path = PARQUET_DIR / "members_v3.parquet"
    TRANSACTIONS_PARQUET: Path = PARQUET_DIR / "transactions.parquet"
    USER_LOGS_PARQUET: Path = PARQUET_DIR / "user_logs.parquet"

    # Processed outputs
    SPINE_PARQUET: Path = PROCESSED_DIR / "spine.parquet"
    TXN_FEATURES_PARQUET: Path = PROCESSED_DIR / "txn_features.parquet"
    LOG_FEATURES_PARQUET: Path = PROCESSED_DIR / "log_features.parquet"


@dataclass(frozen=True)
class Settings:
    # DuckDB temp directory (optional; DuckDB will choose defaults if None)
    DUCKDB_TEMP_DIR: str | None = None

    # General
    RANDOM_SEED: int = 42

    # Members cleaning
    MIN_AGE: int = 10
    MAX_AGE: int = 100


PATHS = Paths()
SETTINGS = Settings()


def ensure_dirs() -> None:
    """Create required directories if they do not exist."""
    PATHS.RAW_DIR.mkdir(parents=True, exist_ok=True)
    PATHS.PARQUET_DIR.mkdir(parents=True, exist_ok=True)
    PATHS.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
