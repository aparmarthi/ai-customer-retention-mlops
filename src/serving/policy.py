"""Decision policy: turns churn scores into actions.

The API loads `artifacts/champion/deployment_policy_v2.json` and calls `assign_action`
for every score, so the tiers live in a versioned artifact rather than in code.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

# PSI rule-of-thumb bands used in credit-risk and churn monitoring.
PSI_STABLE = 0.1
PSI_SHIFT = 0.25
MIN_DRIFT_BATCH = 1000


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
    """Load a deployment policy JSON artifact."""
    return json.loads(policy_path.read_text())


def apply_threshold(prob: float, threshold: float) -> PolicyDecision:
    """Single-record threshold decision."""
    action = "target" if prob >= threshold else "no_target"
    return PolicyDecision(
        churn_probability=float(prob),
        action=action,
        policy_used="threshold",
        threshold=float(threshold),
        metadata={"reason": "probability >= threshold"},
    )


def apply_topk_to_batch(probs: np.ndarray, k: int) -> np.ndarray:
    """Return ranks (1 = highest risk) for a batch of scores."""
    order = np.argsort(-probs)  # descending
    ranks = np.empty_like(order)
    ranks[order] = np.arange(1, len(probs) + 1)
    return ranks


def assign_action(prob: float, policy: dict) -> str:
    """Return the action of the first tier whose min_score the score reaches.

    Args:
        prob: Churn score in [0, 1].
        policy: A v2 policy document with `tiers` sorted by descending `min_score`.

    Returns:
        Tier action, e.g. "contact_now", "nudge" or "monitor".
    """
    for tier in policy["tiers"]:
        if prob >= tier["min_score"]:
            return tier["action"]
    return policy["tiers"][-1]["action"]


def score_drift(probs: np.ndarray, policy: dict, min_n: int = MIN_DRIFT_BATCH) -> Optional[Dict[str, Any]]:
    """Population stability index of a batch's tier mix vs the validation window.

    Output drift needs no labels: if the share of subscribers landing in each tier moves
    away from validation, either the population or the model's behaviour has changed.

    Args:
        probs: Batch churn scores.
        policy: v2 policy document with `validation_summary.tier_counts`.
        min_n: Smaller batches return None because the shares are too noisy.

    Returns:
        {"psi", "status", "expected_share", "observed_share"} or None.
    """
    if len(probs) < min_n:
        return None
    counts = policy["validation_summary"]["tier_counts"]
    total = sum(counts.values())
    actions = [assign_action(float(p), policy) for p in probs]
    psi = 0.0
    expected, observed = {}, {}
    for name, n in counts.items():
        e = n / total
        o = max(actions.count(name) / len(actions), 1e-6)  # avoid log(0) for empty tiers
        psi += (o - e) * math.log(o / e)
        expected[name], observed[name] = round(e, 4), round(o, 4)
    status = "stable" if psi < PSI_STABLE else "moderate_shift" if psi < PSI_SHIFT else "significant_shift"
    return {"psi": round(psi, 4), "status": status, "expected_share": expected, "observed_share": observed}

