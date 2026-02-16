from __future__ import annotations

import json
import sys
import traceback
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from pytorch_tabnet.tab_model import TabNetClassifier
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR

# Add project root to import path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.config import PATHS
from src.utils.run_logger import RunLogger


# -----------------------------
# Single source of truth config
# -----------------------------
TABNET_CONFIG = {
    "random_state": 42,
    "test_size": 0.2,
    "threshold": 0.50,
    # Training runtime knobs (fast verdict)
    "max_epochs": 60,
    "patience": 10,
    "batch_size": 4096,
    "virtual_batch_size": 256,
    "lr": 0.02,
    # OOM fallback candidates (first entry can be overwritten by config above)
    "oom_retries": [(4096, 256), (2048, 256), (1024, 128)],
    # Model architecture
    "n_d": 32,
    "n_a": 32,
    "n_steps": 5,
    "gamma": 1.5,
    "n_independent": 2,
    "n_shared": 2,
    "lambda_sparse": 1e-4,
    "cat_emb_dim": 8,
    "mask_type": "entmax",
    "scheduler_step_size": 20,
    "scheduler_gamma": 0.9,
    "verbose": 5,
}


@dataclass
class TabNetResults:
    n_rows: int
    n_features_raw: int
    churn_rate_overall: float
    churn_rate_train: float
    churn_rate_valid: float
    threshold: float
    accuracy: float
    precision: float
    recall: float
    f1: float
    roc_auc: float
    pr_auc: float
    confusion_matrix: list[list[int]]
    best_epoch: int | None


def _convert_datetime_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Convert datetime64 columns to seconds since epoch (safe; no deprecated .view)."""
    for c in df.columns:
        if np.issubdtype(df[c].dtype, np.datetime64):
            df[c] = (df[c].astype("int64") // 10**9).astype("float32")
    return df


def _clean_numerics(df: pd.DataFrame) -> pd.DataFrame:
    """TabNet can choke on NaN/Inf. Clean aggressively for DL stability."""
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.fillna(0)
    return df


def _fit_cat_mapping(X_train: pd.DataFrame) -> tuple[list[int], list[int], dict[int, dict[Any, int]]]:
    """
    Fit categorical mapping on TRAIN only.

    Returns:
      cat_idxs: indices of categorical columns
      cat_dims_train: number of categories in TRAIN for each categorical column (K)
      cat_maps: mapping col_idx -> {category_value: code} where code in [0..K-1]
    """
    cat_idxs: list[int] = []
    cat_dims_train: list[int] = []
    cat_maps: dict[int, dict[Any, int]] = {}

    for idx, col in enumerate(X_train.columns):
        if str(X_train[col].dtype) in ("object", "category", "bool"):
            s = X_train[col].astype("string").fillna("__NA__")
            cats = pd.Index(s.unique())
            mapping = {cat: i for i, cat in enumerate(cats)}  # 0..K-1
            cat_idxs.append(idx)
            cat_dims_train.append(len(cats))
            cat_maps[idx] = mapping

    return cat_idxs, cat_dims_train, cat_maps


def _apply_tabnet_encoding(
    X: pd.DataFrame,
    cat_idxs: list[int],
    cat_maps: dict[int, dict[Any, int]],
) -> np.ndarray:
    """
    Apply train-fitted mapping to any split.

    Unknown categories -> -1, then shift by +1 so unknown becomes 0.
    Known categories become 1..K. This requires cat_dims to be K+1.
    """
    Xc = X.copy()

    for idx in cat_idxs:
        col = Xc.columns[idx]
        m = cat_maps[idx]
        s = Xc[col].astype("string").fillna("__NA__")
        codes = s.map(m).fillna(-1).astype("int32")
        Xc[col] = (codes + 1).astype("int32")  # unknown=0, known=1..K

    return Xc.to_numpy(dtype=np.float32, copy=False)


def _sanity_check_categoricals(X_np: np.ndarray, cat_idxs: list[int], cat_dims: list[int]) -> None:
    """Ensure categorical columns are within embedding bounds [0..cat_dim-1]."""
    for j, idx in enumerate(cat_idxs):
        mx = int(np.max(X_np[:, idx]))
        mn = int(np.min(X_np[:, idx]))
        if mn < 0 or mx >= cat_dims[j]:
            raise ValueError(
                f"Categorical encoding out of bounds for col_idx={idx}: "
                f"min={mn}, max={mx}, cat_dim={cat_dims[j]}"
            )


def _print_cuda_info() -> None:
    if not torch.cuda.is_available():
        print("CUDA available: False | device: cpu")
        return
    try:
        name = torch.cuda.get_device_name(0)
        mem_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"CUDA available: True | device: {name} | VRAM: {mem_gb:.1f} GB")
    except Exception:
        print("CUDA available: True | device: (unknown)")


def _is_oom_error(e: Exception) -> bool:
    msg = str(e).lower()
    return ("out of memory" in msg) or ("cuda" in msg and "memory" in msg)


def train_tabnet_once(
    X_train_np: np.ndarray,
    y_train: np.ndarray,
    X_valid_np: np.ndarray,
    y_valid: np.ndarray,
    *,
    cat_idxs: list[int],
    cat_dims: list[int],
    pos_weight: float,
    cfg: dict[str, Any],
    batch_size: int,
    virtual_batch_size: int,
) -> TabNetClassifier:
    tabnet_params = dict(
        n_d=cfg["n_d"],
        n_a=cfg["n_a"],
        n_steps=cfg["n_steps"],
        gamma=cfg["gamma"],
        n_independent=cfg["n_independent"],
        n_shared=cfg["n_shared"],
        lambda_sparse=cfg["lambda_sparse"],
        cat_idxs=cat_idxs if cat_idxs else None,
        cat_dims=cat_dims if cat_idxs else None,
        cat_emb_dim=cfg["cat_emb_dim"],
        optimizer_fn=Adam,
        optimizer_params=dict(lr=cfg["lr"]),
        scheduler_fn=StepLR,
        scheduler_params=dict(step_size=cfg["scheduler_step_size"], gamma=cfg["scheduler_gamma"]),
        mask_type=cfg["mask_type"],
        verbose=cfg["verbose"],
        seed=cfg["random_state"],
    )

    model = TabNetClassifier(**{k: v for k, v in tabnet_params.items() if v is not None})

    # Use class weights (TabNetClassifier outputs 2 logits for binary classification)
    class_weights = {0: 1.0, 1: float(pos_weight)}

    model.fit(
        X_train_np,
        y_train,
        eval_set=[(X_valid_np, y_valid)],
        eval_name=["valid"],
        eval_metric=["auc"],  # pytorch-tabnet doesn't support aucpr built-in
        max_epochs=cfg["max_epochs"],
        patience=cfg["patience"],
        batch_size=batch_size,
        virtual_batch_size=virtual_batch_size,
        num_workers=0,
        drop_last=False,
        weights=class_weights,
    )
    return model


def train_tabnet(cfg: dict[str, Any]) -> tuple[TabNetResults, dict[str, Any], TabNetClassifier]:
    model_table_path = PATHS.PROCESSED_DIR / "model_table.parquet"
    if not model_table_path.exists():
        raise FileNotFoundError(f"Missing model table: {model_table_path}")

    print(f"Loading: {model_table_path}")
    df = pd.read_parquet(model_table_path)

    required_cols = {"msno", "is_churn"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"model_table is missing required columns: {missing}")

    y = df["is_churn"].astype(int).values
    X = df.drop(columns=["is_churn", "msno"], errors="ignore").copy()

    X = _convert_datetime_columns(X)
    X = _clean_numerics(X)

    n_rows = len(df)
    n_features_raw = X.shape[1]

    X_train_df, X_valid_df, y_train, y_valid = train_test_split(
        X,
        y,
        test_size=cfg["test_size"],
        random_state=cfg["random_state"],
        stratify=y,
    )

    churn_rate_overall = float(np.mean(y))
    churn_rate_train = float(np.mean(y_train))
    churn_rate_valid = float(np.mean(y_valid))

    _print_cuda_info()

    # Imbalance weight: neg/pos on TRAIN
    pos = float(np.sum(y_train == 1))
    neg = float(np.sum(y_train == 0))
    pos_weight = (neg / pos) if pos > 0 else 1.0
    print(f"pos_weight (neg/pos) = {pos_weight:.2f}")

    # Categorical mapping on TRAIN (+unknown bucket)
    cat_idxs, cat_dims_train, cat_maps = _fit_cat_mapping(X_train_df)
    cat_dims = [d + 1 for d in cat_dims_train]

    X_train_np = _apply_tabnet_encoding(X_train_df, cat_idxs, cat_maps)
    X_valid_np = _apply_tabnet_encoding(X_valid_df, cat_idxs, cat_maps)

    print(f"Rows: {n_rows:,}")
    print(f"Features: {n_features_raw} | categorical={len(cat_idxs)}")
    if cat_idxs:
        cat_cols = [X_train_df.columns[i] for i in cat_idxs]
        print(f"Categorical cols: {cat_cols}")

    if cat_idxs:
        _sanity_check_categoricals(X_train_np, cat_idxs, cat_dims)
        _sanity_check_categoricals(X_valid_np, cat_idxs, cat_dims)

    # OOM retries: start with cfg batch sizes, then fall back list
    retry_plan = [(cfg["batch_size"], cfg["virtual_batch_size"])] + list(cfg["oom_retries"])
    seen: set[tuple[int, int]] = set()

    model: TabNetClassifier | None = None
    last_exc: Exception | None = None
    used_bs, used_vbs = cfg["batch_size"], cfg["virtual_batch_size"]

    for bs, vbs in retry_plan:
        if (bs, vbs) in seen:
            continue
        seen.add((bs, vbs))

        print(f"\nTraining TabNet with batch_size={bs}, virtual_batch_size={vbs}, lr={cfg['lr']} ...")
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            model = train_tabnet_once(
                X_train_np,
                y_train,
                X_valid_np,
                y_valid,
                cat_idxs=cat_idxs,
                cat_dims=cat_dims,
                pos_weight=pos_weight,
                cfg=cfg,
                batch_size=bs,
                virtual_batch_size=vbs,
            )
            used_bs, used_vbs = bs, vbs
            break
        except Exception as e:
            last_exc = e
            print("\n❌ TabNet training failed.")
            traceback.print_exc()

            if _is_oom_error(e) and torch.cuda.is_available():
                print("⚠️ Looks like CUDA OOM. Retrying with smaller batch size...")
                continue
            raise

    if model is None:
        assert last_exc is not None
        raise RuntimeError(f"TabNet failed after retries. Last error: {repr(last_exc)}")

    best_epoch = getattr(model, "best_epoch", None)

    # Predict + metrics
    y_proba = model.predict_proba(X_valid_np)[:, 1]
    y_pred = (y_proba >= cfg["threshold"]).astype(int)

    acc = float(accuracy_score(y_valid, y_pred))
    prec = float(precision_score(y_valid, y_pred, zero_division=0))
    rec = float(recall_score(y_valid, y_pred, zero_division=0))
    f1 = float(f1_score(y_valid, y_pred, zero_division=0))

    try:
        roc = float(roc_auc_score(y_valid, y_proba))
    except ValueError:
        roc = float("nan")

    try:
        pr = float(average_precision_score(y_valid, y_proba))
    except ValueError:
        pr = float("nan")

    cm = confusion_matrix(y_valid, y_pred).tolist()

    results = TabNetResults(
        n_rows=n_rows,
        n_features_raw=int(n_features_raw),
        churn_rate_overall=churn_rate_overall,
        churn_rate_train=churn_rate_train,
        churn_rate_valid=churn_rate_valid,
        threshold=float(cfg["threshold"]),
        accuracy=acc,
        precision=prec,
        recall=rec,
        f1=f1,
        roc_auc=roc,
        pr_auc=pr,
        confusion_matrix=cm,
        best_epoch=int(best_epoch) if best_epoch is not None else None,
    )

    artifacts: dict[str, Any] = {
        "model_table": str(model_table_path),
        "pos_weight": float(pos_weight),
        "categorical_cols": [X_train_df.columns[i] for i in cat_idxs] if cat_idxs else [],
        "cat_dims": cat_dims,
        "batch_size_used": used_bs,
        "virtual_batch_size_used": used_vbs,
        "cfg": {k: v for k, v in cfg.items() if k != "oom_retries"},
    }

    return results, artifacts, model


def main() -> None:
    reports_dir = Path("reports")
    artifacts_dir = Path("artifacts")
    reports_dir.mkdir(parents=True, exist_ok=True)
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    logger = RunLogger("reports/experiment_runs.jsonl")

    cfg = dict(TABNET_CONFIG)  # copy

    def train_fn():
        try:
            results, extra_artifacts, model = train_tabnet(cfg)
        except Exception:
            print("\n❌ TabNet run failed with exception:")
            traceback.print_exc()
            raise

        model_path = artifacts_dir / "tabnet_model"
        model.save_model(str(model_path))  # saves as tabnet_model.zip

        metrics_path = reports_dir / "tabnet_metrics.json"
        metrics_path.write_text(json.dumps(asdict(results), indent=2), encoding="utf-8")

        print("\n=== TabNet Results ===")
        print(f"ROC-AUC: {results.roc_auc:.4f}")
        print(f"PR-AUC:  {results.pr_auc:.4f}")
        print(f"Precision/Recall/F1: {results.precision:.4f} / {results.recall:.4f} / {results.f1:.4f}")
        print("Confusion matrix [[TN, FP],[FN, TP]]:")
        print(results.confusion_matrix)
        if results.best_epoch is not None:
            print(f"Best epoch: {results.best_epoch}")

        return {
            "roc_auc": results.roc_auc,
            "pr_auc": results.pr_auc,
            "accuracy": results.accuracy,
            "precision": results.precision,
            "recall": results.recall,
            "f1": results.f1,
            "threshold": results.threshold,
            "best_epoch": results.best_epoch,
            "n_rows": results.n_rows,
            "n_features_raw": results.n_features_raw,
            "confusion_matrix": results.confusion_matrix,
            "artifact_model": str(model_path) + ".zip",
            "artifact_metrics": str(metrics_path),
            **{f"extra_{k}": v for k, v in extra_artifacts.items()},
        }

    record = logger.log_run(
        model_name="tabnet",
        train_fn=train_fn,
        dataset={"name": "kkbox", "split": f"train/valid (test_size={cfg['test_size']})"},
        params={k: v for k, v in cfg.items() if k != "oom_retries"},
        notes="TabNet hardened & de-duplicated config: NaN/Inf cleanup, train-fitted categorical mapping (+unknown bucket), CUDA OOM retries, class weights, eval_metric=auc.",
    )

    print(f"\n🧾 Logged run: {record['run_id']} ({record['status']})")


if __name__ == "__main__":
    main()
