from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Paths:
    # Project root: …/AI-CUSTOMER-RETENTION-MLOPS/
    ROOT: Path = Path(__file__).resolve().parents[2]

    # Dataset root
    DATA_DIR: Path = ROOT / "data"
    KKBOX_DIR: Path = DATA_DIR / "kkbox"

    # Data layers
    RAW_DIR: Path = KKBOX_DIR / "raw"
    PARQUET_DIR: Path = KKBOX_DIR / "parquet"
    PROCESSED_DIR: Path = KKBOX_DIR / "processed"

    # Raw inputs (KKBox)
    TRAIN_CSV: Path = RAW_DIR / "train.csv"
    MEMBERS_CSV: Path = RAW_DIR / "members_v3.csv"
    TRANSACTIONS_CSV: Path = RAW_DIR / "transactions.csv"
    USER_LOGS_CSV: Path = RAW_DIR / "user_logs.csv"

    # Parquet outputs
    MEMBERS_PARQUET: Path = PARQUET_DIR / "members_v3.parquet"
    TRANSACTIONS_PARQUET: Path = PARQUET_DIR / "transactions.parquet"
    USER_LOGS_PARQUET: Path = PARQUET_DIR / "user_logs.parquet"

    # Processed / feature outputs
    SPINE_PARQUET: Path = PROCESSED_DIR / "spine.parquet"
    TXN_FEATURES_PARQUET: Path = PROCESSED_DIR / "txn_features.parquet"
    LOG_FEATURES_PARQUET: Path = PROCESSED_DIR / "log_features.parquet"


@dataclass(frozen=True)
class Settings:
    # DuckDB
    DUCKDB_TEMP_DIR: str | None = None

    # Reproducibility
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
