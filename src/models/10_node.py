from __future__ import annotations

import json
import sys
import time
import traceback
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from joblib import dump
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
from torch.utils.data import DataLoader, Dataset

# Add project root to import path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.config import PATHS
from src.utils.run_logger import RunLogger


# -----------------------------
# Single source of truth config
# -----------------------------
NODE_CONFIG: Dict[str, Any] = {
    "random_state": 42,
    "test_size": 0.2,
    "threshold": 0.50,

    # NODE hyperparams
    "n_trees": 128,
    "depth": 6,
    "temperature": 10.0,
    "dropout": 0.0,

    # Training
    "batch_size": 4096,
    "lr": 3e-3,
    "weight_decay": 1e-5,
    "max_epochs": 30,
    "patience": 5,
    "grad_clip": 1.0,

    # Performance
    "use_amp": True,
    "num_workers": 0,     # Windows friendly
    "pin_memory": True,   # only helps on CUDA
    "print_every": 1,

    # OOM fallback (CUDA only)
    "oom_batch_retries": [4096, 2048, 1024, 512],
}


# -----------------------------
# Results
# -----------------------------
@dataclass
class NODEResults:
    n_rows: int
    n_features: int
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
    best_epoch: int


# -----------------------------
# Dataset
# -----------------------------
class NumpyDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self) -> int:
        return int(self.y.shape[0])

    def __getitem__(self, idx: int):
        return self.X[idx], self.y[idx]


# -----------------------------
# NODE-style oblivious ensemble
# -----------------------------
class ObliviousTreeEnsemble(nn.Module):
    """
    NODE-inspired differentiable oblivious tree ensemble.

    Each tree has depth D.
    At each depth, the tree chooses a feature (soft selection) and a threshold.
    Split prob = sigmoid((x_sel - thr) * temperature)
    Leaf probs are built by iterative expansion of path probabilities.
    """

    def __init__(
        self,
        n_features: int,
        n_trees: int = 128,
        depth: int = 6,
        temperature: float = 10.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.n_features = n_features
        self.n_trees = n_trees
        self.depth = depth
        self.n_leaves = 2**depth
        self.temperature = temperature
        self.dropout = nn.Dropout(dropout) if dropout > 0 else None

        # (T, D, F)
        self.feature_logits = nn.Parameter(torch.randn(n_trees, depth, n_features) * 0.01)
        # (T, D)
        self.thresholds = nn.Parameter(torch.zeros(n_trees, depth))
        # (T, 2^D)
        self.leaf_values = nn.Parameter(torch.randn(n_trees, self.n_leaves) * 0.01)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.dropout is not None:
            x_in = self.dropout(x)
        else:
            x_in = x

        B, _F = x_in.shape
        T, D = self.n_trees, self.depth

        # (T, D, F)
        w = torch.softmax(self.feature_logits, dim=-1)

        # x_sel: (B, T, D)
        x_sel = torch.einsum("bf,tdf->btd", x_in, w)

        thr = self.thresholds.unsqueeze(0)  # (1, T, D)
        s = torch.sigmoid((x_sel - thr) * self.temperature)  # (B, T, D)

        probs = torch.ones((B, T, 1), device=x.device, dtype=x.dtype)
        for d in range(D):
            sd = s[:, :, d : d + 1]
            probs = torch.cat([probs * (1.0 - sd), probs * sd], dim=-1)

        leaf = self.leaf_values.unsqueeze(0)  # (1, T, L)
        tree_out = (probs * leaf).sum(dim=-1)  # (B, T)
        logits = tree_out.sum(dim=1)  # (B,)
        return logits


# -----------------------------
# Preprocessing
# -----------------------------
def _set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _convert_datetime_columns(df: pd.DataFrame) -> pd.DataFrame:
    for c in df.columns:
        if np.issubdtype(df[c].dtype, np.datetime64):
            df[c] = (df[c].astype("int64") // 10**9).astype("float32")
    return df


def _clean_numerics(df: pd.DataFrame) -> pd.DataFrame:
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.fillna(0)
    return df


def _fit_onehot_train_apply_valid(
    X_train: pd.DataFrame, X_valid: pd.DataFrame, cat_cols: list[str], num_cols: list[str]
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """
    Train-fitted one-hot for categoricals to avoid leakage / inconsistent columns.

    We build dummies on TRAIN, then align VALID to TRAIN columns.
    """
    if cat_cols:
        tr_cat = pd.get_dummies(
            X_train[cat_cols].astype("string").fillna("__NA__"),
            prefix=cat_cols,
            dummy_na=False,
        )
        va_cat = pd.get_dummies(
            X_valid[cat_cols].astype("string").fillna("__NA__"),
            prefix=cat_cols,
            dummy_na=False,
        )
        va_cat = va_cat.reindex(columns=tr_cat.columns, fill_value=0)

        X_train_all = pd.concat([X_train[num_cols], tr_cat], axis=1)
        X_valid_all = pd.concat([X_valid[num_cols], va_cat], axis=1)
    else:
        X_train_all = X_train[num_cols].copy()
        X_valid_all = X_valid[num_cols].copy()

    # Standardize all columns (critical for thresholds)
    tr = X_train_all.to_numpy(dtype=np.float32, copy=False)
    va = X_valid_all.to_numpy(dtype=np.float32, copy=False)

    mean = tr.mean(axis=0, keepdims=True)
    std = tr.std(axis=0, keepdims=True)
    std = np.where(std < 1e-6, 1.0, std)

    tr = (tr - mean) / std
    va = (va - mean) / std

    meta = {
        "cat_cols": cat_cols,
        "num_cols": num_cols,
        "onehot_cols": list(X_train_all.columns),
        "mean": mean.astype(np.float32),
        "std": std.astype(np.float32),
    }
    return tr.astype(np.float32), va.astype(np.float32), meta


@torch.no_grad()
def _predict_proba(model: nn.Module, loader: DataLoader, device: torch.device, use_amp: bool) -> np.ndarray:
    model.eval()
    probs: list[np.ndarray] = []
    for xb, _yb in loader:
        xb = xb.to(device, non_blocking=True)
        if device.type == "cuda" and use_amp:
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                logits = model(xb)
        else:
            logits = model(xb)
        p = torch.sigmoid(logits).detach().cpu().numpy()
        probs.append(p)
    return np.concatenate(probs, axis=0)


def _is_oom_error(e: Exception) -> bool:
    msg = str(e).lower()
    return ("out of memory" in msg) or ("cuda" in msg and "memory" in msg)


# -----------------------------
# Train/eval
# -----------------------------
def train_node(cfg: dict[str, Any]) -> Tuple[NODEResults, Dict[str, Any], Dict[str, Any]]:
    _set_seed(cfg["random_state"])

    model_table_path = PATHS.PROCESSED_DIR / "model_table.parquet"
    if not model_table_path.exists():
        raise FileNotFoundError(f"Missing model table: {model_table_path}")

    print(f"Loading: {model_table_path}")
    df = pd.read_parquet(model_table_path)

    required_cols = {"msno", "is_churn"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"model_table is missing required columns: {missing}")

    y = df["is_churn"].astype(int).to_numpy()
    X = df.drop(columns=["is_churn", "msno"], errors="ignore").copy()

    X = _convert_datetime_columns(X)
    X = _clean_numerics(X)

    # Split raw DF first (avoid leakage in standardization/one-hot)
    X_train_df, X_valid_df, y_train, y_valid = train_test_split(
        X,
        y,
        test_size=cfg["test_size"],
        random_state=cfg["random_state"],
        stratify=y,
    )

    # Identify cols
    cat_cols = [c for c in X_train_df.columns if str(X_train_df[c].dtype) in ("object", "category", "bool")]
    num_cols = [c for c in X_train_df.columns if c not in cat_cols]

    # Train-fitted one-hot + standardization
    X_train, X_valid, preprocess_meta = _fit_onehot_train_apply_valid(X_train_df, X_valid_df, cat_cols, num_cols)

    churn_rate_overall = float(np.mean(y))
    churn_rate_train = float(np.mean(y_train))
    churn_rate_valid = float(np.mean(y_valid))

    pos = float(np.sum(y_train == 1))
    neg = float(np.sum(y_train == 0))
    pos_weight = (neg / pos) if pos > 0 else 1.0

    print(f"Rows: {len(df):,} | Features(after onehot): {X_train.shape[1]}")
    print(f"Categorical cols: {len(cat_cols)} | pos_weight (neg/pos) = {pos_weight:.2f}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        name = torch.cuda.get_device_name(0)
        vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"Device: cuda | {name} | VRAM: {vram_gb:.1f} GB")
    else:
        print("Device: cpu")

    pin_memory = bool(cfg["pin_memory"] and device.type == "cuda")

    def make_loaders(batch_size: int) -> tuple[DataLoader, DataLoader]:
        train_ds = NumpyDataset(X_train, y_train)
        valid_ds = NumpyDataset(X_valid, y_valid)
        train_loader = DataLoader(
            train_ds,
            batch_size=batch_size,
            shuffle=True,
            num_workers=cfg["num_workers"],
            pin_memory=pin_memory,
        )
        valid_loader = DataLoader(
            valid_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=cfg["num_workers"],
            pin_memory=pin_memory,
        )
        return train_loader, valid_loader

    model = ObliviousTreeEnsemble(
        n_features=X_train.shape[1],
        n_trees=cfg["n_trees"],
        depth=cfg["depth"],
        temperature=cfg["temperature"],
        dropout=cfg["dropout"],
    ).to(device)

    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight], device=device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    scaler_amp = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda" and cfg["use_amp"]))

    def run_with_batch(batch_size: int) -> Tuple[NODEResults, Dict[str, Any], Dict[str, Any]]:
        train_loader, valid_loader = make_loaders(batch_size)

        best_pr = -1.0
        best_epoch = -1
        best_state = None
        bad_epochs = 0

        for epoch in range(1, cfg["max_epochs"] + 1):
            t0 = time.time()
            model.train()
            total_loss = 0.0

            for xb, yb in train_loader:
                xb = xb.to(device, non_blocking=True)
                yb = yb.to(device, non_blocking=True)

                optimizer.zero_grad(set_to_none=True)

                if device.type == "cuda" and cfg["use_amp"]:
                    with torch.autocast(device_type="cuda", dtype=torch.float16):
                        logits = model(xb)
                        loss = criterion(logits, yb)
                    scaler_amp.scale(loss).backward()
                    scaler_amp.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
                    scaler_amp.step(optimizer)
                    scaler_amp.update()
                else:
                    logits = model(xb)
                    loss = criterion(logits, yb)
                    loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
                    optimizer.step()

                total_loss += float(loss.item()) * xb.size(0)

            y_proba = _predict_proba(model, valid_loader, device=device, use_amp=cfg["use_amp"])
            pr = float(average_precision_score(y_valid, y_proba))
            avg_loss = total_loss / len(train_loader.dataset)
            dt = time.time() - t0

            if epoch % cfg["print_every"] == 0:
                print(f"Epoch {epoch:02d} | loss={avg_loss:.5f} | valid_pr_auc={pr:.5f} | time={dt:.1f}s")

            if pr > best_pr + 1e-5:
                best_pr = pr
                best_epoch = epoch
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                bad_epochs = 0
            else:
                bad_epochs += 1
                if bad_epochs >= cfg["patience"]:
                    print(f"Early stopping at epoch {epoch} (best epoch={best_epoch}, best PR-AUC={best_pr:.5f})")
                    break

        if best_state is not None:
            model.load_state_dict(best_state)

        # Final evaluation
        y_proba = _predict_proba(model, valid_loader, device=device, use_amp=cfg["use_amp"])
        y_pred = (y_proba >= cfg["threshold"]).astype(int)

        acc = float(accuracy_score(y_valid, y_pred))
        prec = float(precision_score(y_valid, y_pred, zero_division=0))
        rec = float(recall_score(y_valid, y_pred, zero_division=0))
        f1 = float(f1_score(y_valid, y_pred, zero_division=0))

        try:
            roc = float(roc_auc_score(y_valid, y_proba))
        except ValueError:
            roc = float("nan")

        cm = confusion_matrix(y_valid, y_pred).tolist()

        results = NODEResults(
            n_rows=len(df),
            n_features=int(X_train.shape[1]),
            churn_rate_overall=churn_rate_overall,
            churn_rate_train=churn_rate_train,
            churn_rate_valid=churn_rate_valid,
            threshold=float(cfg["threshold"]),
            accuracy=acc,
            precision=prec,
            recall=rec,
            f1=f1,
            roc_auc=roc,
            pr_auc=float(best_pr),
            confusion_matrix=cm,
            best_epoch=int(best_epoch),
        )

        artifacts = {
            "model_table": str(model_table_path),
            "pos_weight": float(pos_weight),
            "preprocess": preprocess_meta,
            "node_params": {
                "n_trees": cfg["n_trees"],
                "depth": cfg["depth"],
                "temperature": cfg["temperature"],
                "dropout": cfg["dropout"],
            },
            "batch_size_used": batch_size,
            "device": str(device),
            "cfg": cfg,
        }

        bundle = {
            "state_dict": model.state_dict(),
            "config": {
                "n_features": int(X_train.shape[1]),
                "n_trees": cfg["n_trees"],
                "depth": cfg["depth"],
                "temperature": cfg["temperature"],
                "dropout": cfg["dropout"],
            },
        }

        return results, artifacts, bundle

    # OOM-safe batch plan
    batch_plan = [cfg["batch_size"]] + [b for b in cfg["oom_batch_retries"] if b != cfg["batch_size"]]
    last_exc: Exception | None = None

    for bsz in batch_plan:
        try:
            if device.type == "cuda":
                torch.cuda.empty_cache()
            print(f"\nTraining NODE with batch_size={bsz}, amp={cfg['use_amp'] and device.type=='cuda'} ...")
            return run_with_batch(bsz)
        except Exception as e:
            last_exc = e
            print("\n❌ NODE training failed.")
            traceback.print_exc()
            if device.type == "cuda" and _is_oom_error(e):
                print("⚠️ CUDA OOM. Retrying with smaller batch size...")
                continue
            raise

    assert last_exc is not None
    raise RuntimeError(f"NODE failed after retries. Last error: {repr(last_exc)}")


def main() -> None:
    reports_dir = Path("reports")
    artifacts_dir = Path("artifacts")
    reports_dir.mkdir(parents=True, exist_ok=True)
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    logger = RunLogger("reports/experiment_runs.jsonl")
    cfg = dict(NODE_CONFIG)

    def train_fn():
        try:
            results, artifacts, bundle = train_node(cfg)
        except Exception:
            print("\n❌ NODE run failed with exception:")
            traceback.print_exc()
            raise

        model_path = artifacts_dir / "node_bundle.joblib"
        dump({"bundle": bundle, "artifacts": artifacts}, model_path)

        metrics_path = reports_dir / "node_metrics.json"
        metrics_path.write_text(json.dumps(asdict(results), indent=2), encoding="utf-8")

        print("\n=== NODE-style Oblivious Ensemble Results ===")
        print(f"ROC-AUC: {results.roc_auc:.4f}")
        print(f"PR-AUC:  {results.pr_auc:.4f}")
        print(f"Precision/Recall/F1: {results.precision:.4f} / {results.recall:.4f} / {results.f1:.4f}")
        print("Confusion matrix [[TN, FP],[FN, TP]]:")
        print(results.confusion_matrix)
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
            "n_features": results.n_features,
            "confusion_matrix": results.confusion_matrix,
            "artifact_model": str(model_path),
            "artifact_metrics": str(metrics_path),
            "batch_size_used": artifacts.get("batch_size_used"),
            "device": artifacts.get("device"),
        }

    record = logger.log_run(
        model_name="node-oblivious",
        train_fn=train_fn,
        dataset={"name": "kkbox", "split": f"train/valid (test_size={cfg['test_size']})"},
        params=cfg,
        notes="NODE optimized: single config, train-fitted one-hot, standardization, NaN/Inf cleanup, AMP+GradScaler, OOM batch fallback, early stop on PR-AUC.",
    )

    print(f"\n🧾 Logged run: {record['run_id']} ({record['status']})")


if __name__ == "__main__":
    main()
