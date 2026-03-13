# Champion Bundle Notes (FLAML AutoML — LightGBM)

## What this folder is

This folder contains the **frozen "champion" churn model** and the minimal artifacts needed to reproduce scoring + targeting decisions consistently.

| File | Purpose |
|---|---|
| `model.pkl` | Trained champion model (FLAML best estimator: LightGBM) |
| `threshold.json` | Decision policy parameters (ROI-optimal threshold + ops top-K) |
| `deployment_policy.json` | Hybrid policy definition (primary: top-K, fallback: threshold) |
| `metrics.json` | Complete evaluation summary |
| `feature_list.json` | Exact feature columns + order used at train/score time |
| `flaml_best_params.json` | Full FLAML-tuned hyperparameter configuration |
| `roi_policy.json` | ROI simulation assumptions and results |
| `valid_scored.parquet` | Scored validation set (`msno`, `y_true`, `y_proba`) |
| `notes.md` | This file |

---

## Dataset / Label

- **Dataset:** KKBox churn prediction dataset (subscription behavior + transactions + user activity features)
- **Label:** `is_churn` (binary)

---

## Split Strategy

- **Split key:** `log_last_date` (latest observed user activity date), stored as `YYYYMMDD` numeric and parsed using `pd.to_datetime(..., format="%Y%m%d")`. Note: `metrics.json` records this as `txn_last_date` — same column, aliased during processing.
- **Date coverage:** 2015-01-01 to 2017-02-28
- **Cutoff policy:** `quantile_0.8` — the training script uses an 80th-percentile date cutoff, which resolves to approximately Jan 31, 2017 on the full dataset
- **Train:** rows where `log_last_date <= cutoff` OR `log_last_date == 0` (missing/no-logs)
- **Validation:** rows where `log_last_date > cutoff` and `log_last_date != 0`
- **Training window:** 2015-01-01 through ~2017-01-31 (inclusive), plus records with no log history
- **Validation window:** ~2017-02-01 through 2017-02-28

---

## Features

- Feature set is fixed via `feature_list.json`
- Do not add/remove/reorder columns unless retraining a new champion
- Identifiers (e.g., `msno`) are excluded from training features (kept only for joins / reporting)

---

## Model

- **Architecture:** LightGBM (selected by FLAML AutoML)
- **Key hyperparameters:** `num_leaves=1212`, `learning_rate=0.0358`, `n_estimators=146`, `reg_alpha=0.56`
- **Goal:** Maximize ranking quality for churn targeting under severe class imbalance

---

## Champion Metrics (time-based holdout)

| Metric | Value |
|---|---:|
| ROC-AUC | 0.9660 |
| PR-AUC | **0.5392** |
| F1 @ 0.5 | 0.3678 |
| Precision @ top-5k | 0.2690 |
| Precision @ top-10k | **0.1801** |
| Recall @ top-10k | 0.7498 |
| Recall @ top-20k | 0.9072 |
| Validation churn rate | 1.24% |

---

## Decision Policy

The production policy is **hybrid** — see `threshold.json` and `deployment_policy.json`:

| Policy | Type | Key Parameter | Use Case |
|---|---|---|---|
| **Primary** | Top-K | K = 10,000 | Ops-driven: predictable contact volume, budget-bounded |
| **Fallback** | ROI-optimal threshold | t = 0.68 | Cost-sensitive: automated low-cost interventions |

**ROI-optimal threshold details:**
- Contacts: 1,478
- Precision: 70.6%
- Estimated net ROI: $17,666 (under documented assumptions in `reports/business_assumptions.md`)

---

## Reproducibility Checklist

To keep this "champion" stable:

1. Use the exact same processed dataset version used for the run
2. Score using the exact columns and order in `feature_list.json`
3. Apply the policy from `threshold.json` for binary actioning
4. If retraining is required, retrain deterministically using:
   - `flaml_best_params.json`
   - Fixed random seed
   - Same time split definition (cutoff = 20170131)

---

## Known Limitations

- **Label delay:** In production, churn labels arrive ~30 days after scoring. During that window, only input drift signals provide early warning
- **Calibration:** Model probabilities are not calibrated — scores are used for ranking (top-K) and threshold comparison, not as true probability estimates
- **Single holdout month:** Validation covers Feb 2017 only. A rolling-window backtest across multiple months would provide more robust performance estimates
