from __future__ import annotations

import json
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
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

# -----------------------------------------------------------------------------
# Project root / imports
# -----------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.config import PATHS  # noqa: E402
from src.utils.run_logger import RunLogger  # noqa: E402

# -----------------------------------------------------------------------------
# Single source of truth config
# -----------------------------------------------------------------------------
ENSEMBLE_CFG = {
    "random_state": 42,
    "test_size": 0.2,

    # Threshold behavior
    "threshold": 0.50,
    "tune_threshold": False,   # set True to tune threshold on validation
    "threshold_grid": [i / 100 for i in range(5, 96, 1)],  # 0.05..0.95

    # Weight behavior
    # mode: "manual" | "equal" | "prauc_proportional" | "grid_search"
    "weight_mode": "manual",

    # Used when weight_mode="manual"
    "manual_weights": {
        "lightgbm": 1.0,
        # "xgboost": 1.0,
        # "tabnet": 1.0,
        # "catboost": 1.0,
    },

    # Used when weight_mode="grid_search" (2-3 models recommended)
    # Search weights for each model in this list (sum to 1)
    "grid_weight_step": 0.05,  # 0.05 => 21 steps, reasonable

    # Probability files (must be predicted on SAME valid split)
    # NOTE: these are relative to PROJECT_ROOT and will be resolved safely.
    "probs_files": {
        "lightgbm": "reports/preds_lightgbm_valid.npy",
        # "xgboost": "reports/preds_xgboost_valid.npy",
        # "tabnet": "reports/preds_tabnet_valid.npy",
        # "catboost": "reports/preds_catboost_valid.npy",
    },
}


# -----------------------------------------------------------------------------
# Results schema
# -----------------------------------------------------------------------------
@dataclass
class EnsembleResults:
    threshold: float
    roc_auc: float
    pr_auc: float
    accuracy: float
    precision: float
    recall: float
    f1: float
    confusion_matrix: list[list[int]]
    models: list[str]
    weights: dict[str, float]
    per_model_pr_auc: dict[str, float]
    per_model_roc_auc: dict[str, float]


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def _load_model_table_split(random_state: int, test_size: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Recreate y_valid exactly the same way as your other scripts (stratified).
    Assumes your canonical table is PATHS.PROCESSED_DIR / "model_table.parquet"
    and label column is "is_churn".
    """
    model_table_path = PATHS.PROCESSED_DIR / "model_table.parquet"
    df = pd.read_parquet(model_table_path)

    if "is_churn" not in df.columns:
        raise KeyError("Expected label column 'is_churn' in model_table.parquet")

    y = df["is_churn"].astype(int).values
    idx = np.arange(len(df))

    _idx_train, idx_valid, _y_train, y_valid = train_test_split(
        idx, y, test_size=test_size, random_state=random_state, stratify=y
    )
    return idx_valid, y_valid


def _load_probs(path: Path) -> np.ndarray:
    """Load probability vector from .npy or .csv."""
    if not path.exists():
        raise FileNotFoundError(
            "Missing probability file:\n"
            f"  expected: {path}\n"
            f"  cwd:      {Path.cwd()}\n"
            "  hint: ensure your base model saved preds to reports/, or fix ENSEMBLE_CFG paths."
        )

    if path.suffix.lower() == ".npy":
        arr = np.load(path)
        return arr.astype(float)

    if path.suffix.lower() == ".csv":
        df = pd.read_csv(path)
        for col in ("proba", "y_proba", "prob", "p"):
            if col in df.columns:
                return df[col].to_numpy(dtype=float)
        raise ValueError(f"{path} must contain a probability column like 'proba' or 'y_proba'.")

    raise ValueError(f"Unsupported file type: {path.suffix}. Use .npy or .csv.")


def _metrics(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    threshold: float,
) -> tuple[float, float, float, float, float, float, list[list[int]]]:
    """Compute standard classification metrics."""
    y_pred = (y_proba >= threshold).astype(int)

    roc = float(roc_auc_score(y_true, y_proba))
    pr = float(average_precision_score(y_true, y_proba))
    acc = float(accuracy_score(y_true, y_pred))
    prec = float(precision_score(y_true, y_pred, zero_division=0))
    rec = float(recall_score(y_true, y_pred, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))
    cm = confusion_matrix(y_true, y_pred).tolist()

    return roc, pr, acc, prec, rec, f1, cm


def _softvote(probs: Dict[str, np.ndarray], weights: Dict[str, float]) -> np.ndarray:
    """Weighted average of probability vectors."""
    keys = list(probs.keys())
    w_sum = sum(weights[k] for k in keys)
    if w_sum <= 0:
        raise ValueError("Weights must sum to a positive number.")

    y_ens = np.zeros_like(next(iter(probs.values())), dtype=float)
    for k in keys:
        y_ens += (weights[k] / w_sum) * probs[k]
    return y_ens


def _normalize_weights(w: Dict[str, float]) -> Dict[str, float]:
    s = float(sum(w.values()))
    if s <= 0:
        raise ValueError("Weights must sum to a positive number.")
    return {k: float(v) / s for k, v in w.items()}


def _choose_weights(
    mode: str,
    probs: Dict[str, np.ndarray],
    y_valid: np.ndarray,
    cfg: dict,
) -> Dict[str, float]:
    """Select weights based on cfg mode."""
    keys = list(probs.keys())

    if mode == "equal":
        return {k: 1.0 for k in keys}

    if mode == "manual":
        w = dict(cfg.get("manual_weights", {}))
        # ensure all present keys have weights
        for k in keys:
            w.setdefault(k, 1.0)
        # drop weights for models not present
        w = {k: float(w[k]) for k in keys}
        return w

    if mode == "prauc_proportional":
        w = {}
        for k in keys:
            w[k] = float(average_precision_score(y_valid, probs[k]))
        # floor to avoid zeros
        w = {k: max(v, 1e-6) for k, v in w.items()}
        return w

    if mode == "grid_search":
        # Keep search tractable
        if len(keys) < 2 or len(keys) > 3:
            raise ValueError("grid_search mode supports 2 or 3 models only (for tractable search).")

        step = float(cfg.get("grid_weight_step", 0.05))
        if step <= 0 or step > 1:
            raise ValueError("grid_weight_step must be in (0, 1].")

        grid = np.arange(0.0, 1.0 + 1e-9, step)
        best_w: Dict[str, float] | None = None
        best_pr = -1.0

        if len(keys) == 2:
            k1, k2 = keys
            for a in grid:
                w = {k1: float(a), k2: float(1.0 - a)}
                y_ens = _softvote(probs, w)
                pr = float(average_precision_score(y_valid, y_ens))
                if pr > best_pr:
                    best_pr = pr
                    best_w = w
        else:
            k1, k2, k3 = keys
            for a in grid:
                for b in grid:
                    c = 1.0 - a - b
                    if c < -1e-9:
                        continue
                    if c < 0:
                        c = 0.0
                    s = a + b + c
                    if s <= 0:
                        continue
                    w = {k1: float(a), k2: float(b), k3: float(c)}
                    y_ens = _softvote(probs, w)
                    pr = float(average_precision_score(y_valid, y_ens))
                    if pr > best_pr:
                        best_pr = pr
                        best_w = w

        if best_w is None:
            raise RuntimeError("No valid weights found in grid search.")
        return best_w

    raise ValueError(f"Unknown weight_mode: {mode}")


def _tune_threshold(y_valid: np.ndarray, y_proba: np.ndarray, grid: list[float]) -> float:
    """Tune threshold to maximize F1 on the validation set."""
    best_t = 0.5
    best_f1 = -1.0
    for t in grid:
        y_pred = (y_proba >= t).astype(int)
        f1 = float(f1_score(y_valid, y_pred, zero_division=0))
        if f1 > best_f1:
            best_f1 = f1
            best_t = float(t)
    return float(best_t)


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main() -> None:
    cfg = dict(ENSEMBLE_CFG)

    random_state = int(cfg["random_state"])
    test_size = float(cfg["test_size"])
    default_threshold = float(cfg["threshold"])

    _idx_valid, y_valid = _load_model_table_split(random_state=random_state, test_size=test_size)

    # Resolve and load probability files from repo root (robust to cwd)
    probs: Dict[str, np.ndarray] = {}
    print("\nLooking for probability files:")
    for name, rel_path in cfg["probs_files"].items():
        path = (PROJECT_ROOT / rel_path).resolve()
        print(f"  {name}: {path}  exists={path.exists()}")
        arr = _load_probs(path)

        if len(arr) != len(y_valid):
            raise ValueError(
                f"{name} probs length {len(arr)} != y_valid length {len(y_valid)}. "
                "Make sure you saved preds on the same valid split."
            )
        probs[name] = arr

    if len(probs) < 1:
        raise ValueError("No probability files configured in ENSEMBLE_CFG['probs_files'].")

    # Per-model metrics (for logging + sanity)
    per_model_pr = {k: float(average_precision_score(y_valid, probs[k])) for k in probs.keys()}
    per_model_roc = {k: float(roc_auc_score(y_valid, probs[k])) for k in probs.keys()}

    # Choose weights
    weights_raw = _choose_weights(cfg["weight_mode"], probs, y_valid, cfg)
    weights = _normalize_weights(weights_raw)

    # Soft-vote ensemble
    y_ens = _softvote(probs, weights)

    # Optional threshold tuning (maximize F1)
    threshold = default_threshold
    if bool(cfg.get("tune_threshold", False)):
        threshold = _tune_threshold(y_valid, y_ens, cfg["threshold_grid"])

    # Final metrics
    roc, pr, acc, prec, rec, f1, cm = _metrics(y_valid, y_ens, threshold)

    results = EnsembleResults(
        threshold=threshold,
        roc_auc=roc,
        pr_auc=pr,
        accuracy=acc,
        precision=prec,
        recall=rec,
        f1=f1,
        confusion_matrix=cm,
        models=list(probs.keys()),
        weights=weights,
        per_model_pr_auc=per_model_pr,
        per_model_roc_auc=per_model_roc,
    )

    print("\n=== Soft-Vote Ensemble ===")
    print(f"Models: {results.models}")
    print(f"Per-model PR-AUC: {results.per_model_pr_auc}")
    print(f"Weights (normalized): {results.weights}")
    print(f"Threshold: {results.threshold:.2f}")
    print(f"ROC-AUC: {results.roc_auc:.4f}")
    print(f"PR-AUC:  {results.pr_auc:.4f}")
    print(f"Precision/Recall/F1: {results.precision:.4f} / {results.recall:.4f} / {results.f1:.4f}")
    print("Confusion matrix [[TN, FP],[FN, TP]]:")
    print(results.confusion_matrix)

    # Save metrics to reports/ under project root
    reports_dir = (PROJECT_ROOT / "reports").resolve()
    reports_dir.mkdir(parents=True, exist_ok=True)
    out_path = reports_dir / "ensemble_softvote_metrics.json"
    out_path.write_text(json.dumps(asdict(results), indent=2), encoding="utf-8")
    print(f"\n✅ Saved: {out_path}")

    # Log run
    logger = RunLogger(str(reports_dir / "experiment_runs.jsonl"))
    record = logger.log_run(
        model_name="ensemble-softvote",
        train_fn=lambda: {
            "roc_auc": results.roc_auc,
            "pr_auc": results.pr_auc,
            "precision": results.precision,
            "recall": results.recall,
            "f1": results.f1,
            "accuracy": results.accuracy,
            "threshold": results.threshold,
            "confusion_matrix": results.confusion_matrix,
            "models": results.models,
            "weights": results.weights,
            "per_model_pr_auc": results.per_model_pr_auc,
            "per_model_roc_auc": results.per_model_roc_auc,
            "artifact_metrics": str(out_path),
        },
        dataset={"name": "kkbox", "split": f"valid (test_size={test_size})"},
        params={k: v for k, v in cfg.items() if k != "probs_files"},
        notes="Soft-voting ensemble over base model probability outputs (with optional weight search + threshold tuning).",
    )

    print(f"\n🧾 Logged run: {record['run_id']} ({record['status']})")


if __name__ == "__main__":
    main()
