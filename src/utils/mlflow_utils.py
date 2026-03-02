"""
mlflow_utils.py

Minimal MLflow wrapper utilities for local-first experiment tracking.

Logs:
- Params: model hyperparams, time_col, cutoff policy, feature version
- Metrics: PR-AUC, ROC-AUC, F1, Precision@K, Recall@K
- Artifacts: metrics.json, plots, shap plot, confusion matrix image, etc.

Usage (typical):
    from src.utils.mlflow_utils import (
        start_run,
        log_params_flat,
        log_metrics_safe,
        log_artifacts_safe,
        set_experiment,
        end_run,
    )

    set_experiment("kkbox_churn")
    with start_run(run_name="champion_lgbm_time_split"):
        log_params_flat({...})
        log_metrics_safe({...})
        log_artifacts_safe([Path("artifacts/champion/metrics.json"), Path("reports/threshold_vs_roi.png")])
"""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Union

import json

import mlflow


# -----------------------------
# Core helpers
# -----------------------------

def set_experiment(experiment_name: str) -> None:
    """Set (or create) an MLflow experiment."""
    mlflow.set_experiment(experiment_name)


def start_run(
    run_name: Optional[str] = None,
    tags: Optional[Dict[str, Any]] = None,
):
    """
    Start an MLflow run (context-manager friendly).

    Example:
        with start_run("run1", tags={"stage":"champion"}):
            ...
    """
    run = mlflow.start_run(run_name=run_name)
    if tags:
        # MLflow expects stringable values
        mlflow.set_tags({k: _stringify(v) for k, v in tags.items()})
    return run


def end_run(status: str = "FINISHED") -> None:
    """End the active MLflow run."""
    mlflow.end_run(status=status)


# -----------------------------
# Param logging
# -----------------------------

def log_params_flat(params: Dict[str, Any], prefix: str = "") -> None:
    """
    Log parameters to MLflow safely, flattening nested dicts.

    - Nested keys become 'a.b.c'
    - Values are stringified if needed

    MLflow has a param length limit; this keeps it simple and robust.
    """
    flat = _flatten_dict(params)
    if prefix:
        flat = {f"{prefix}{k}": v for k, v in flat.items()}

    # MLflow params must be key -> string
    safe_params = {k: _stringify(v) for k, v in flat.items()}
    mlflow.log_params(safe_params)


# -----------------------------
# Metric logging
# -----------------------------

def log_metrics_safe(metrics: Dict[str, Any], step: Optional[int] = None, prefix: str = "") -> None:
    """
    Log numeric metrics safely.

    - Filters out non-numeric values
    - Optionally prefixes keys
    """
    safe: Dict[str, float] = {}
    for k, v in metrics.items():
        key = f"{prefix}{k}" if prefix else k
        fv = _to_float_or_none(v)
        if fv is not None:
            safe[key] = fv

    if not safe:
        return

    mlflow.log_metrics(safe, step=step)


# -----------------------------
# Artifact logging
# -----------------------------

def log_artifacts_safe(
    paths: Iterable[Union[str, Path]],
    artifact_path: Optional[str] = None,
) -> None:
    """
    Log files or directories as MLflow artifacts.

    - If a path is a file -> log_artifact
    - If a path is a dir  -> log_artifacts
    - Skips missing paths
    """
    for p in paths:
        path = Path(p)
        if not path.exists():
            continue

        if path.is_dir():
            mlflow.log_artifacts(str(path), artifact_path=artifact_path)
        else:
            mlflow.log_artifact(str(path), artifact_path=artifact_path)


def log_json_artifact(
    obj: Any,
    out_path: Union[str, Path],
    artifact_path: Optional[str] = None,
    indent: int = 2,
) -> Path:
    """
    Write a JSON file locally and log it to MLflow.
    Useful for metrics.json, threshold.json, params snapshot, etc.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if is_dataclass(obj):
        payload = asdict(obj)
    else:
        payload = obj

    out_path.write_text(json.dumps(payload, indent=indent))
    mlflow.log_artifact(str(out_path), artifact_path=artifact_path)
    return out_path


# -----------------------------
# Optional convenience: common churn metrics bundle
# -----------------------------

def build_eval_metrics_dict(
    *,
    pr_auc: float,
    roc_auc: float,
    f1: float,
    precision_at_k: Optional[Dict[int, float]] = None,
    recall_at_k: Optional[Dict[int, float]] = None,
) -> Dict[str, float]:
    """
    Standardize naming and flatten dicts for MLflow logging.

    Returns keys like:
      pr_auc, roc_auc, f1
      precision_at_5000, precision_at_10000
      recall_at_5000, recall_at_10000
    """
    out: Dict[str, float] = {
        "pr_auc": float(pr_auc),
        "roc_auc": float(roc_auc),
        "f1": float(f1),
    }

    if precision_at_k:
        for k, v in precision_at_k.items():
            out[f"precision_at_{int(k)}"] = float(v)

    if recall_at_k:
        for k, v in recall_at_k.items():
            out[f"recall_at_{int(k)}"] = float(v)

    return out


# -----------------------------
# Internal utilities
# -----------------------------

def _flatten_dict(d: Dict[str, Any], parent_key: str = "", sep: str = ".") -> Dict[str, Any]:
    items: Dict[str, Any] = {}
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else str(k)
        if isinstance(v, dict):
            items.update(_flatten_dict(v, new_key, sep=sep))
        else:
            items[new_key] = v
    return items


def _stringify(v: Any) -> str:
    if v is None:
        return "None"
    if isinstance(v, (str, int, float, bool)):
        return str(v)
    # handle lists/tuples/small dicts
    try:
        return json.dumps(v, default=str)
    except Exception:
        return str(v)


def _to_float_or_none(v: Any) -> Optional[float]:
    if v is None:
        return None
    if isinstance(v, (int, float, np_number())):
        try:
            return float(v)
        except Exception:
            return None
    # allow strings that parse
    if isinstance(v, str):
        try:
            return float(v)
        except Exception:
            return None
    return None


def np_number():
    """Avoid importing numpy as a hard dependency here; support numpy scalar floats/ints if present."""
    try:
        import numpy as np  # local import
        return (np.integer, np.floating)
    except Exception:
        return tuple()