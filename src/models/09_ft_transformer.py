from __future__ import annotations

import json
import math
import sys
import time
import traceback
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

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
FTT_CONFIG: Dict[str, Any] = {
    "random_state": 42,
    "test_size": 0.2,
    "threshold": 0.50,

    # Model
    "d_token": 192,
    "n_heads": 8,
    "n_layers": 4,
    "dropout": 0.1,
    "ffn_mult": 4,

    # Training
    "batch_size": 4096,
    "lr": 3e-4,
    "weight_decay": 1e-5,
    "max_epochs": 30,
    "patience": 5,
    "grad_clip": 1.0,

    # Performance
    "use_amp": True,            # mixed precision on GPU
    "num_workers": 0,           # Windows friendly
    "pin_memory": True,         # only helps when CUDA
    "prefetch_factor": None,    # keep None for Windows/num_workers=0
    "print_every": 1,           # log each epoch

    # OOM fallback (CUDA only)
    "oom_batch_retries": [4096, 2048, 1024, 512],
}


# -----------------------------
# Results dataclass
# -----------------------------
@dataclass
class FTTransformerResults:
    n_rows: int
    n_num_features: int
    n_cat_features: int
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
class TabularDataset(Dataset):
    def __init__(self, X_num: np.ndarray, X_cat: np.ndarray, y: np.ndarray):
        self.X_num = torch.tensor(X_num, dtype=torch.float32)
        self.X_cat = torch.tensor(X_cat, dtype=torch.long)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self) -> int:
        return int(self.y.shape[0])

    def __getitem__(self, idx: int):
        return self.X_num[idx], self.X_cat[idx], self.y[idx]


# -----------------------------
# FT-Transformer model
# -----------------------------
class FTTransformer(nn.Module):
    def __init__(
        self,
        n_num: int,
        cat_cardinalities: list[int],
        d_token: int = 192,
        n_heads: int = 8,
        n_layers: int = 4,
        dropout: float = 0.1,
        ffn_mult: int = 4,
    ):
        super().__init__()
        self.n_num = n_num
        self.cat_cardinalities = cat_cardinalities
        self.n_cat = len(cat_cardinalities)
        self.d_token = d_token

        # CLS token
        self.cls = nn.Parameter(torch.zeros(1, 1, d_token))

        # Numeric tokenization: per-feature affine map into token space
        self.num_weight = nn.Parameter(torch.randn(n_num, d_token) / math.sqrt(d_token))
        self.num_bias = nn.Parameter(torch.zeros(n_num, d_token))

        # Categorical embeddings
        self.cat_embs = nn.ModuleList([nn.Embedding(card, d_token) for card in cat_cardinalities])

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_token,
            nhead=n_heads,
            dim_feedforward=ffn_mult * d_token,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        self.norm = nn.LayerNorm(d_token)
        self.head = nn.Sequential(
            nn.Linear(d_token, d_token),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_token, 1),
        )

        nn.init.normal_(self.cls, mean=0.0, std=0.02)

    def forward(self, x_num: torch.Tensor, x_cat: torch.Tensor) -> torch.Tensor:
        # Numeric tokens: (B, n_num, d_token)
        num_tokens = x_num.unsqueeze(-1) * self.num_weight.unsqueeze(0) + self.num_bias.unsqueeze(0)

        # Categorical tokens: (B, n_cat, d_token) or None
        if self.n_cat > 0:
            cat_tokens = torch.stack([emb(x_cat[:, i]) for i, emb in enumerate(self.cat_embs)], dim=1)
        else:
            cat_tokens = None

        # [CLS] + numeric + categorical
        B = x_num.shape[0]
        cls = self.cls.expand(B, -1, -1)
        tokens = torch.cat([cls, num_tokens] + ([cat_tokens] if cat_tokens is not None else []), dim=1)

        h = self.encoder(tokens)
        cls_out = self.norm(h[:, 0, :])
        logits = self.head(cls_out).squeeze(-1)
        return logits


# -----------------------------
# Utilities
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


def _standardize_train_valid(
    X_train: np.ndarray, X_valid: np.ndarray
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    mean = X_train.mean(axis=0, keepdims=True)
    std = X_train.std(axis=0, keepdims=True)
    std = np.where(std < 1e-6, 1.0, std)
    X_train_s = (X_train - mean) / std
    X_valid_s = (X_valid - mean) / std
    scaler = {"mean": mean.astype(np.float32), "std": std.astype(np.float32)}
    return X_train_s.astype(np.float32), X_valid_s.astype(np.float32), scaler


def _encode_categoricals_train_valid(
    X_train_cat: pd.DataFrame, X_valid_cat: pd.DataFrame
) -> tuple[np.ndarray, np.ndarray, list[int], dict[str, Any]]:
    """
    Ordinal encode categoricals using TRAIN mapping.
    Unknown categories -> 0.
    Known categories -> 1..K
    cat_cardinalities -> K+1 (unknown bucket included)
    """
    mappings: dict[str, dict[str, int]] = {}
    train_codes = np.zeros((len(X_train_cat), X_train_cat.shape[1]), dtype=np.int64)
    valid_codes = np.zeros((len(X_valid_cat), X_valid_cat.shape[1]), dtype=np.int64)
    cardinalities: list[int] = []

    for j, col in enumerate(X_train_cat.columns):
        tr = X_train_cat[col].astype("string").fillna("__NA__")
        va = X_valid_cat[col].astype("string").fillna("__NA__")

        uniq = pd.Index(tr.unique())
        to_id = {v: i + 1 for i, v in enumerate(uniq)}  # 1..K
        mappings[col] = to_id
        cardinalities.append(len(uniq) + 1)  # +1 for unknown bucket (0)

        train_codes[:, j] = tr.map(to_id).fillna(0).astype(np.int64).to_numpy()
        valid_codes[:, j] = va.map(to_id).fillna(0).astype(np.int64).to_numpy()

    return train_codes, valid_codes, cardinalities, mappings


@torch.no_grad()
def _predict_proba(model: nn.Module, loader: DataLoader, device: torch.device, use_amp: bool) -> np.ndarray:
    model.eval()
    probs: list[np.ndarray] = []
    for x_num, x_cat, _y in loader:
        x_num = x_num.to(device, non_blocking=True)
        x_cat = x_cat.to(device, non_blocking=True)
        if device.type == "cuda" and use_amp:
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                logits = model(x_num, x_cat)
        else:
            logits = model(x_num, x_cat)
        p = torch.sigmoid(logits).detach().cpu().numpy()
        probs.append(p)
    return np.concatenate(probs, axis=0)


def _is_oom_error(e: Exception) -> bool:
    msg = str(e).lower()
    return ("out of memory" in msg) or ("cuda" in msg and "memory" in msg)


def train_ft_transformer(cfg: dict[str, Any]) -> tuple[FTTransformerResults, dict[str, Any], dict[str, Any]]:
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

    # Split
    X_train_df, X_valid_df, y_train, y_valid = train_test_split(
        X,
        y,
        test_size=cfg["test_size"],
        random_state=cfg["random_state"],
        stratify=y,
    )

    # Columns
    cat_cols = [c for c in X_train_df.columns if str(X_train_df[c].dtype) in ("object", "category", "bool")]
    num_cols = [c for c in X_train_df.columns if c not in cat_cols]

    print(f"Rows: {len(df):,}")
    print(f"Numeric cols: {len(num_cols)} | Categorical cols: {len(cat_cols)}")
    if cat_cols:
        print(f"Categorical columns: {cat_cols}")

    # Standardize numerics
    X_train_num = X_train_df[num_cols].to_numpy(dtype=np.float32, copy=False)
    X_valid_num = X_valid_df[num_cols].to_numpy(dtype=np.float32, copy=False)
    X_train_num, X_valid_num, scaler = _standardize_train_valid(X_train_num, X_valid_num)

    # Encode categoricals (train mapping + unknown bucket)
    if cat_cols:
        tr_cat, va_cat, cat_cardinalities, cat_mappings = _encode_categoricals_train_valid(
            X_train_df[cat_cols], X_valid_df[cat_cols]
        )
    else:
        tr_cat = np.zeros((len(X_train_df), 0), dtype=np.int64)
        va_cat = np.zeros((len(X_valid_df), 0), dtype=np.int64)
        cat_cardinalities = []
        cat_mappings = {}

    churn_rate_overall = float(np.mean(y))
    churn_rate_train = float(np.mean(y_train))
    churn_rate_valid = float(np.mean(y_valid))

    pos = float(np.sum(y_train == 1))
    neg = float(np.sum(y_train == 0))
    pos_weight = (neg / pos) if pos > 0 else 1.0
    print(f"pos_weight (neg/pos) = {pos_weight:.2f}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        name = torch.cuda.get_device_name(0)
        vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"Device: cuda | {name} | VRAM: {vram_gb:.1f} GB")
    else:
        print("Device: cpu")

    # DataLoaders (pin_memory helps only on GPU)
    pin_memory = bool(cfg["pin_memory"] and device.type == "cuda")

    def make_loaders(batch_size: int) -> tuple[DataLoader, DataLoader]:
        train_ds = TabularDataset(X_train_num, tr_cat, y_train)
        valid_ds = TabularDataset(X_valid_num, va_cat, y_valid)

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

    # Build model
    model = FTTransformer(
        n_num=len(num_cols),
        cat_cardinalities=cat_cardinalities,
        d_token=cfg["d_token"],
        n_heads=cfg["n_heads"],
        n_layers=cfg["n_layers"],
        dropout=cfg["dropout"],
        ffn_mult=cfg["ffn_mult"],
    ).to(device)

    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight], device=device))

    def run_with_batch(batch_size: int) -> tuple[FTTransformerResults, dict[str, Any], dict[str, Any]]:
        train_loader, valid_loader = make_loaders(batch_size)

        optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
        scaler_amp = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda" and cfg["use_amp"]))

        best_pr = -1.0
        best_epoch = -1
        best_state = None
        bad_epochs = 0

        for epoch in range(1, cfg["max_epochs"] + 1):
            t0 = time.time()
            model.train()
            total_loss = 0.0

            for x_num, x_cat, yb in train_loader:
                x_num = x_num.to(device, non_blocking=True)
                x_cat = x_cat.to(device, non_blocking=True)
                yb = yb.to(device, non_blocking=True)

                optimizer.zero_grad(set_to_none=True)

                if device.type == "cuda" and cfg["use_amp"]:
                    with torch.autocast(device_type="cuda", dtype=torch.float16):
                        logits = model(x_num, x_cat)
                        loss = criterion(logits, yb)
                    scaler_amp.scale(loss).backward()
                    scaler_amp.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
                    scaler_amp.step(optimizer)
                    scaler_amp.update()
                else:
                    logits = model(x_num, x_cat)
                    loss = criterion(logits, yb)
                    loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
                    optimizer.step()

                total_loss += float(loss.item()) * x_num.size(0)

            # Validation PR-AUC
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

        # Final eval
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

        results = FTTransformerResults(
            n_rows=len(df),
            n_num_features=len(num_cols),
            n_cat_features=len(cat_cols),
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
            "num_cols": num_cols,
            "cat_cols": cat_cols,
            "cat_cardinalities": cat_cardinalities,
            "pos_weight": float(pos_weight),
            "scaler": scaler,
            "cat_mappings": cat_mappings,
            "batch_size_used": batch_size,
            "device": str(device),
            "cfg": cfg,
        }

        bundle = {
            "state_dict": model.state_dict(),
            "config": {
                "n_num": len(num_cols),
                "cat_cardinalities": cat_cardinalities,
                "d_token": cfg["d_token"],
                "n_heads": cfg["n_heads"],
                "n_layers": cfg["n_layers"],
                "dropout": cfg["dropout"],
                "ffn_mult": cfg["ffn_mult"],
            },
        }

        return results, artifacts, bundle

    # OOM-safe training: try configured batch size then smaller ones on CUDA
    batch_plan = [cfg["batch_size"]] + [b for b in cfg["oom_batch_retries"] if b != cfg["batch_size"]]
    last_exc: Exception | None = None

    for bsz in batch_plan:
        try:
            if device.type == "cuda":
                torch.cuda.empty_cache()
            print(f"\nTraining FT-Transformer with batch_size={bsz}, amp={cfg['use_amp'] and device.type=='cuda'} ...")
            return run_with_batch(bsz)
        except Exception as e:
            last_exc = e
            print("\n❌ FT-Transformer training failed.")
            traceback.print_exc()
            if device.type == "cuda" and _is_oom_error(e):
                print("⚠️ CUDA OOM. Retrying with smaller batch size...")
                continue
            raise

    assert last_exc is not None
    raise RuntimeError(f"FT-Transformer failed after retries. Last error: {repr(last_exc)}")


def main() -> None:
    reports_dir = Path("reports")
    artifacts_dir = Path("artifacts")
    reports_dir.mkdir(parents=True, exist_ok=True)
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    logger = RunLogger("reports/experiment_runs.jsonl")
    cfg = dict(FTT_CONFIG)  # copy

    def train_fn():
        try:
            results, extra, bundle = train_ft_transformer(cfg)
        except Exception:
            print("\n❌ FT-Transformer run failed with exception:")
            traceback.print_exc()
            raise

        model_path = artifacts_dir / "ft_transformer_bundle.joblib"
        dump(
            {
                "bundle": bundle,
                "preprocess": {
                    "num_cols": extra["num_cols"],
                    "cat_cols": extra["cat_cols"],
                    "scaler": extra["scaler"],
                    "cat_mappings": extra["cat_mappings"],
                    "cat_cardinalities": extra["cat_cardinalities"],
                },
            },
            model_path,
        )

        metrics_path = reports_dir / "ft_transformer_metrics.json"
        metrics_path.write_text(json.dumps(asdict(results), indent=2), encoding="utf-8")

        print("\n=== FT-Transformer Results ===")
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
            "n_num_features": results.n_num_features,
            "n_cat_features": results.n_cat_features,
            "confusion_matrix": results.confusion_matrix,
            "artifact_model": str(model_path),
            "artifact_metrics": str(metrics_path),
            "batch_size_used": extra.get("batch_size_used"),
            "device": extra.get("device"),
        }

    record = logger.log_run(
        model_name="ft-transformer",
        train_fn=train_fn,
        dataset={"name": "kkbox", "split": f"train/valid (test_size={cfg['test_size']})"},
        params=cfg,
        notes="FT-Transformer optimized: single config, NaN/Inf cleanup, train-fitted cat encoding w/ unknown bucket, AMP+GradScaler, OOM batch fallback, early stop on PR-AUC.",
    )

    print(f"\n🧾 Logged run: {record['run_id']} ({record['status']})")


if __name__ == "__main__":
    main()
