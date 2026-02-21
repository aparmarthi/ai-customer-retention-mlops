from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Dict, Any

import numpy as np
import pandas as pd


@dataclass
class PolicyDecision:
    churn_probability: float
    action: str  # "target" or "no_target"
    policy_used: str  # "top_k" or "threshold"
    threshold: Optional[float] = None
    top_k: Optional[int] = None
    rank: Optional[int] = None
    metadata: Optional[Dict[str, Any]] = None


def load_policy(policy_path: Path) -> dict:
    return json.loads(policy_path.read_text())


def apply_threshold(prob: float, threshold: float) -> PolicyDecision:
    action = "target" if prob >= threshold else "no_target"
    return PolicyDecision(
        churn_probability=float(prob),
        action=action,
        policy_used="threshold",
        threshold=float(threshold),
        metadata={"reason": "probability >= threshold"}
    )


def apply_topk_to_batch(probs: np.ndarray, k: int) -> np.ndarray:
    """
    Returns an array of ranks (1 = highest risk). Lower rank is higher risk.
    """
    order = np.argsort(-probs)  # descending
    ranks = np.empty_like(order)
    ranks[order] = np.arange(1, len(probs) + 1)
    return ranks


def decide_single_with_fallback(prob: float, policy: dict) -> PolicyDecision:
    """
    For single-record prediction, top-K is not meaningful (needs batch ranking).
    So we fall back to threshold policy.
    """
    thr = float(policy["secondary_policy"]["threshold"])
    d = apply_threshold(prob, thr)
    d.metadata = d.metadata or {}
    d.metadata["note"] = "Single prediction: top-K requires batch context; using threshold fallback."
    return d


def decide_batch(df_scored: pd.DataFrame, policy: dict) -> pd.DataFrame:
    """
    df_scored must contain:
      - y_proba: float column
    Adds:
      - rank
      - action
      - policy_used
    """
    probs = df_scored["y_proba"].to_numpy(dtype=float)
    k = int(policy["primary_policy"]["k"])
    thr = float(policy["secondary_policy"]["threshold"])

    ranks = apply_topk_to_batch(probs, k=k)
    df_out = df_scored.copy()
    df_out["rank"] = ranks

    # Primary top-K
    df_out["action"] = np.where(df_out["rank"] <= k, "target", "no_target")
    df_out["policy_used"] = "top_k"

    # If you want to also provide threshold-based action as a column:
    df_out["action_threshold"] = np.where(df_out["y_proba"] >= thr, "target", "no_target")
    df_out["threshold_used"] = thr
    df_out["top_k_used"] = k

    return df_out
