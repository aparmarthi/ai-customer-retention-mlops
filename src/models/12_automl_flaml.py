# src/models/12_automl_flaml.py
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd
from flaml import AutoML
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

# Add project root to import path so `import src...` works
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.config import PATHS  # Paths object in your repo
from src.utils.run_logger import RunLogger  # RunLogger(log_file=...) only


@dataclass
class AutoMLRunConfig:
    time_budget_s: int = 1800  # 30 min
    metric: str = "ap"  # FLAML "ap" = Average Precision (PR-AUC)
    task: str = "classification"
    seed: int = 42
    n_jobs: int = -1

    estimators: Tuple[str, ...] = ("lgbm", "xgboost", "catboost", "lrl2")

    top_k: int = 10000
    threshold: float = 0.5

    cutoff_quantile: float = 0.80  # for cutoff=auto
    min_valid_rows: int = 10_000   # ensure holdout isn't tiny


def resolve_default_data_path() -> Path:
    fallback = PROJECT_ROOT / "data" / "kkbox" / "processed" / "model_table.parquet"

    # dict-like (if supported)
    try:
        kkbox = PATHS["kkbox"]  # type: ignore[index]
        processed = kkbox["processed"]  # type: ignore[index]
        return Path(processed) / "model_table.parquet"
    except Exception:
        pass

    # attribute-like patterns
    try:
        kkbox_obj = getattr(PATHS, "kkbox")
        processed = getattr(kkbox_obj, "processed")
        return Path(processed) / "model_table.parquet"
    except Exception:
        pass

    for attr in ("kkbox_processed", "processed_kkbox"):
        try:
            processed = getattr(PATHS, attr)
            return Path(processed) / "model_table.parquet"
        except Exception:
            pass

    return fallback


def normalize_time_column(df: pd.DataFrame, time_col: str) -> pd.Series:
    """
    Ensure df[time_col] becomes a 1D datetime Series.
    If each cell is list/tuple/np.ndarray, use the max element (latest date).
    """
    s = df[time_col]

    def _to_scalar(x):
        if isinstance(x, (list, tuple, np.ndarray)):
            if len(x) == 0:
                return pd.NaT
            try:
                return max(x)
            except TypeError:
                xx = pd.to_datetime(list(x), errors="coerce")
                return xx.max()
        return x

    s2 = s.map(_to_scalar)
    return pd.to_datetime(s2, errors="coerce")


def compute_topk_metrics(y_true: np.ndarray, y_proba: np.ndarray, k: int) -> Dict[str, float]:
    k = int(min(k, len(y_true)))
    if k <= 0:
        return {"k": 0.0, "precision_at_k": 0.0, "recall_at_k": 0.0}
    idx = np.argsort(-y_proba)[:k]
    y_true_topk = y_true[idx]
    precision_at_k = float(y_true_topk.mean())
    recall_at_k = float(y_true_topk.sum() / max(1, y_true.sum()))
    return {"k": float(k), "precision_at_k": precision_at_k, "recall_at_k": recall_at_k}


def evaluate_binary(y_true: np.ndarray, y_proba: np.ndarray, threshold: float) -> Dict[str, Any]:
    y_pred = (y_proba >= threshold).astype(int)
    return {
        "threshold": float(threshold),
        "roc_auc": float(roc_auc_score(y_true, y_proba)),
        "pr_auc": float(average_precision_score(y_true, y_proba)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }


def parse_args() -> argparse.Namespace:
    default_data_path = resolve_default_data_path()

    p = argparse.ArgumentParser(description="FLAML AutoML for KKBox churn (time-based split)")
    p.add_argument("--data-path", type=str, default=str(default_data_path))
    p.add_argument("--target-col", type=str, default="is_churn")
    p.add_argument("--id-col", type=str, default="msno")
    p.add_argument("--time-col", type=str, required=True)

    # allow "auto"
    p.add_argument(
        "--cutoff",
        type=str,
        required=True,
        help="Time cutoff. Use 'auto' to pick cutoff by quantile (default 0.80).",
    )
    p.add_argument("--cutoff-quantile", type=float, default=0.80)
    p.add_argument("--min-valid-rows", type=int, default=10_000)

    p.add_argument("--time-budget-s", type=int, default=1800)
    p.add_argument("--top-k", type=int, default=10000)
    p.add_argument("--threshold", type=float, default=0.5)
    return p.parse_args()


def pick_auto_cutoff(time_series: pd.Series, quantile: float, min_valid_rows: int) -> pd.Timestamp:
    """
    Pick a cutoff so that roughly `quantile` of rows fall in train.
    If validation would be too small, reduce quantile until valid_rows >= min_valid_rows.
    """
    s = time_series.dropna()
    if s.empty:
        raise ValueError("Time series is empty after dropna; cannot pick cutoff.")

    q = float(quantile)
    q = min(max(q, 0.50), 0.95)

    for _ in range(12):
        cutoff = pd.to_datetime(s.quantile(q))
        valid_rows = int((time_series > cutoff).sum())
        if valid_rows >= int(min_valid_rows):
            return cutoff
        q -= 0.05
        if q < 0.50:
            break

    # fallback: median
    return pd.to_datetime(s.quantile(0.50))


def coerce_cutoff(time_series: pd.Series, cutoff_raw: str, quantile: float, min_valid_rows: int) -> pd.Timestamp:
    """
    Convert cutoff_raw into a pd.Timestamp.
    Supports cutoff_raw='auto'.
    """
    if cutoff_raw.strip().lower() == "auto":
        return pick_auto_cutoff(time_series, quantile=quantile, min_valid_rows=min_valid_rows)
    try:
        return pd.to_datetime(cutoff_raw)
    except Exception as e:
        raise ValueError(f"Could not parse cutoff '{cutoff_raw}' as datetime: {e}") from e


def time_based_split(df: pd.DataFrame, time_col: str, cutoff: pd.Timestamp) -> Tuple[pd.DataFrame, pd.DataFrame]:
    train = df.loc[df[time_col] <= cutoff].copy()
    valid = df.loc[df[time_col] > cutoff].copy()
    return train, valid


def main() -> None:
    args = parse_args()

    cfg = AutoMLRunConfig(
        time_budget_s=args.time_budget_s,
        top_k=args.top_k,
        threshold=args.threshold,
        cutoff_quantile=args.cutoff_quantile,
        min_valid_rows=args.min_valid_rows,
    )

    # init logger (even if we only append jsonl ourselves)
    _ = RunLogger(log_file=str(PROJECT_ROOT / "reports" / "experiment_runs.jsonl"))

    # artifacts dir
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"automl_flaml_{run_id}"
    run_dir = PROJECT_ROOT / "artifacts" / "runs" / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    # load data
    data_path = Path(args.data_path)
    if not data_path.exists():
        raise FileNotFoundError(f"Data path not found: {data_path}")

    df = pd.read_parquet(data_path)

    target_col = args.target_col
    time_col = args.time_col
    id_col = args.id_col

    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found.")
    if time_col not in df.columns:
        candidates = [c for c in df.columns if any(k in c.lower() for k in ("date", "time", "month", "asof"))]
        raise ValueError(f"Time column '{time_col}' not found. Candidates: {candidates[:25]}")

    # normalize + drop invalid
    df[time_col] = normalize_time_column(df, time_col)
    before = len(df)
    df = df.dropna(subset=[time_col]).copy()
    after = len(df)
    if after < before:
        print(f"Dropped {before - after} rows with invalid {time_col} values.")

    # cutoff (supports auto)
    cutoff = coerce_cutoff(df[time_col], args.cutoff, quantile=cfg.cutoff_quantile, min_valid_rows=cfg.min_valid_rows)

    # split
    train_df, valid_df = time_based_split(df, time_col=time_col, cutoff=cutoff)

    # if split is still bad, force auto cutoff
    if len(train_df) == 0 or len(valid_df) == 0:
        cutoff2 = pick_auto_cutoff(df[time_col], quantile=cfg.cutoff_quantile, min_valid_rows=cfg.min_valid_rows)
        train_df, valid_df = time_based_split(df, time_col=time_col, cutoff=cutoff2)
        print(f"[cutoff override] Provided cutoff produced empty split. Using auto cutoff={cutoff2}.")
        cutoff = cutoff2

    print(f"Cutoff used: {cutoff}")
    print(f"Split sizes -> train: {len(train_df):,} | valid: {len(valid_df):,}")

    # features
    drop_cols = [target_col]
    if id_col in df.columns:
        drop_cols.append(id_col)
    feature_cols = [c for c in df.columns if c not in drop_cols]

    X_train = train_df[feature_cols].copy()
    y_train = train_df[target_col].astype(int).to_numpy()

    X_valid = valid_df[feature_cols].copy()
    y_valid = valid_df[target_col].astype(int).to_numpy()

    # object -> category
    obj_cols = [c for c in X_train.columns if X_train[c].dtype == "object"]
    for c in obj_cols:
        X_train[c] = X_train[c].astype("category")
        X_valid[c] = X_valid[c].astype("category")

    # automl
    start = time.time()
    automl = AutoML()
    automl.fit(
        X_train=X_train,
        y_train=y_train,
        time_budget=cfg.time_budget_s,
        metric=cfg.metric,
        task=cfg.task,
        seed=cfg.seed,
        n_jobs=cfg.n_jobs,
        estimator_list=list(cfg.estimators),
    )
    elapsed_s = time.time() - start

    # evaluate
    y_proba = automl.predict_proba(X_valid)[:, 1]
    metrics = evaluate_binary(y_valid, y_proba, threshold=cfg.threshold)
    topk = compute_topk_metrics(y_valid, y_proba, k=cfg.top_k)

    # save artifacts
    import joblib

    joblib.dump(automl, run_dir / "model.joblib")
    (run_dir / "feature_cols.json").write_text(json.dumps(feature_cols, indent=2))
    (run_dir / "threshold_policy.json").write_text(json.dumps({"threshold": cfg.threshold}, indent=2))

    run_summary = {
        "run_name": run_name,
        "run_type": "automl_flaml",
        "timestamp": run_id,
        "data_path": str(data_path),
        "time_col": time_col,
        "cutoff": str(cutoff),
        "target_col": target_col,
        "id_col": id_col if id_col in df.columns else None,
        "n_rows": int(len(df)),
        "train_rows": int(len(train_df)),
        "valid_rows": int(len(valid_df)),
        "feature_count": int(len(feature_cols)),
        "elapsed_s": float(elapsed_s),
        "estimators_allowed": list(cfg.estimators),
        "best_estimator": str(getattr(automl, "best_estimator", None)),
        "best_config": getattr(automl, "best_config", None),
        "best_loss": float(getattr(automl, "best_loss", np.nan))
        if getattr(automl, "best_loss", None) is not None
        else None,
        "metrics": metrics,
        "topk": topk,
        "artifacts_dir": str(run_dir),
    }
    (run_dir / "metrics.json").write_text(json.dumps(run_summary, indent=2))

    # append JSONL record
    reports_dir = PROJECT_ROOT / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = reports_dir / "experiment_runs.jsonl"

    log_record = {
        "run_name": run_name,
        "run_type": "automl_flaml",
        "timestamp": run_id,
        "artifacts_dir": str(run_dir),
        "best_estimator": run_summary["best_estimator"],
        "best_config": run_summary["best_config"],
        "elapsed_s": float(elapsed_s),
        "metrics": {
            "roc_auc": metrics["roc_auc"],
            "pr_auc": metrics["pr_auc"],
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "f1": metrics["f1"],
            "precision_at_k": topk["precision_at_k"],
            "recall_at_k": topk["recall_at_k"],
            "k": topk["k"],
            "threshold": metrics["threshold"],
        },
    }
    with jsonl_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(log_record) + "\n")

    print(f"\n✅ Saved run artifacts to: {run_dir}")
    print(f"✅ Appended run summary to: {jsonl_path}")
    print(f"Best estimator: {run_summary['best_estimator']}")
    print(
        f"Valid PR-AUC: {metrics['pr_auc']:.4f} | ROC-AUC: {metrics['roc_auc']:.4f} | "
        f"F1@{cfg.threshold}: {metrics['f1']:.4f} | P@{int(topk['k'])}: {topk['precision_at_k']:.4f}"
    )


if __name__ == "__main__":
    main()
