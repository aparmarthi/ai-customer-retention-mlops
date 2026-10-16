"""
Tests for the v2 decision policy and its consistency with the champion artifacts.

Run:
  python -m pytest src/serving/test_policy.py -v
"""

import json
from pathlib import Path

import numpy as np
import pytest

from src.serving.policy import assign_action, load_policy, score_drift

ART = Path(__file__).resolve().parents[2] / "artifacts" / "champion"


@pytest.fixture(scope="module")
def policy():
    return load_policy(ART / "deployment_policy_v2.json")


@pytest.fixture(scope="module")
def thresholds():
    return json.loads((ART / "threshold.json").read_text())


def test_tiers_follow_score_cutoffs(policy):
    assert assign_action(0.95, policy) == "contact_now"
    assert assign_action(0.68, policy) == "contact_now"
    assert assign_action(0.30, policy) == "nudge"
    assert assign_action(0.2058, policy) == "nudge"
    assert assign_action(0.01, policy) == "monitor"


def test_cutoffs_match_threshold_artifact(policy, thresholds):
    cutoffs = {t["action"]: t["min_score"] for t in policy["tiers"]}
    assert cutoffs["contact_now"] == thresholds["roi_optimal"]["threshold"]
    assert cutoffs["nudge"] == thresholds["ops_top_k"]["equiv_threshold"]


def test_validation_counts_match_threshold_artifact(policy, thresholds):
    summary = policy["validation_summary"]
    counts, churners = summary["tier_counts"], summary["tier_churners"]
    assert sum(counts.values()) == summary["valid_rows"] == 193205
    assert sum(churners.values()) == summary["churners"] == 2402
    assert counts["contact_now"] == thresholds["roi_optimal"]["n_targeted"]
    assert counts["contact_now"] + counts["nudge"] == thresholds["ops_top_k"]["k"]
    assert churners["contact_now"] == thresholds["roi_optimal"]["tp"]
    assert churners["contact_now"] + churners["nudge"] == thresholds["ops_top_k"]["tp_at_k"]


def test_net_values_recompute_from_tiers(policy):
    """net = churners x save rate x value - contacts x cost, per tier."""
    v = policy["value_per_save"]
    counts = policy["validation_summary"]["tier_counts"]
    churners = policy["validation_summary"]["tier_churners"]
    tiers = {t["action"]: t for t in policy["tiers"]}

    def tier_net(name):
        t = tiers[name]
        return churners[name] * t["assumed_save_rate"] * v - counts[name] * t["cost_per_contact"]

    reported = policy["validation_summary"]["net_per_cycle"]
    assert tier_net("contact_now") == pytest.approx(reported["threshold_only"], abs=1)
    assert tier_net("contact_now") + tier_net("nudge") == pytest.approx(reported["tiered"], abs=1)


def test_drift_is_stable_at_validation_mix(policy):
    counts = policy["validation_summary"]["tier_counts"]
    probs = np.concatenate([
        np.full(counts["contact_now"], 0.9),
        np.full(counts["nudge"], 0.4),
        np.full(counts["monitor"], 0.05),
    ])
    drift = score_drift(probs, policy)
    assert drift["psi"] == pytest.approx(0, abs=1e-6)
    assert drift["status"] == "stable"


def test_drift_flags_a_shifted_mix(policy):
    probs = np.concatenate([np.full(300, 0.9), np.full(300, 0.4), np.full(400, 0.05)])
    assert score_drift(probs, policy)["status"] == "significant_shift"


def test_drift_skips_small_batches(policy):
    assert score_drift(np.full(999, 0.5), policy) is None
