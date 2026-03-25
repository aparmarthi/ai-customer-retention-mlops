# KKBox Churn — Model Leaderboard

Sorted by **PR-AUC (Average Precision)**, then ROC-AUC. Metrics are on the **valid split** (random 80/20 split used for all Phase 2-3 experiments).

| Rank | Model | PR-AUC | PR-AUC Lift | ROC-AUC | F1 | Precision | Recall | Train Time | Notes |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | LightGBM | 0.8887 | **14.3x** | 0.9894 | 0.7845 | 0.6805 | 0.9260 | ~3 min | Baseline config, scale_pos_weight |
| 2 | Ensemble (soft vote) | 0.8887 | **14.3x** | 0.9894 | 0.7845 | 0.6805 | 0.9260 | — | Reduces to LightGBM (weight = 1.0) |
| 3 | XGBoost (Train API) | 0.8771 | **14.1x** | 0.9875 | — | — | — | ~2 min | Baseline XGBoost, no threshold tuning |
| 4 | CatBoost | 0.8737 | **14.1x** | 0.9865 | 0.7282 | 0.6002 | 0.9257 | ~3.7 min | GPU-accelerated, early stop (od_wait=200) |
| 5 | FT-Transformer | 0.8214 | **13.2x** | 0.9824 | 0.6825 | 0.5374 | 0.9350 | ~23 min | 4-layer transformer, AMP, GPU |
| 6 | Random Forest | 0.7935 | **12.8x** | 0.9782 | 0.5798 | 0.4188 | 0.9419 | ~5 min | One-hot encoding, balanced_subsample |
| 7 | NODE | 0.7719 | **12.4x** | 0.9737 | 0.5334 | 0.3717 | 0.9446 | ~11 min | 128 oblivious trees, depth 6, GPU |
| 8 | TabNet | 0.5233 | **8.4x** | 0.9085 | 0.3998 | 0.5671 | 0.3087 | ~32 min | 5 failed runs before convergence |

> **Lift** = PR-AUC / baseline PR-AUC. Baseline PR-AUC ≈ class prevalence (~6.2% on random split). A 14.3x lift means the model ranks churners 14x better than random ordering.

### Champion (time-based holdout — different evaluation)

The champion uses a **time-based holdout** (Feb 2017), not the random split above. These numbers are not directly comparable to the leaderboard — they are harder and more realistic.

| Metric | Champion (FLAML AutoML — LightGBM) | Lift |
|---|---:|---:|
| ROC-AUC | 0.9660 | **1.9x** vs random (0.50) |
| PR-AUC | 0.5392 | **43.5x** vs base rate (1.24%) |
| F1 (at 0.5) | 0.3678 | — |
| P@5k | 0.2690 | **21.7x** vs base rate |
| P@10k | 0.1801 | **14.5x** vs base rate |
| R@5k | 0.5600 | — |
| R@10k | 0.7498 | — |
| FLAML search time | ~30 min | — |

## Champion selection criteria

The FLAML AutoML LightGBM was selected as champion based on:

1. **Highest PR-AUC on time-based holdout (0.5392)** — PR-AUC is the primary metric because it directly measures ranking quality under severe class imbalance (1.24% churn rate in validation)
2. **LightGBM architecture dominance** — LightGBM ranked #1 in the random-split leaderboard (0.8887 PR-AUC) and FLAML's AutoML search independently converged on LightGBM as the best estimator, confirming the architecture choice
3. **FLAML hyperparameter tuning** — FLAML explored a broader hyperparameter space than manual tuning (e.g., `num_leaves=1212` vs. manual `64`, `reg_alpha=0.56`), producing a configuration that generalizes better to the harder time-based split
4. **Training efficiency** — LightGBM trains in ~3 minutes vs. 23-32 minutes for deep learning alternatives (FT-Transformer, TabNet) that scored worse on every metric

## Notes

- **PR-AUC is the primary metric** because churn is highly imbalanced (~6.4% overall, ~1.24% in the time-based holdout). Accuracy and ROC-AUC overstate performance under severe imbalance.
- **Threshold-dependent metrics** (Precision, Recall, F1) assume a fixed 0.5 cutoff. The champion's production threshold is set by ROI-optimized policy — see `artifacts/champion/threshold.json`.
- **Train times** are wall-clock on a single machine (LAPTOP-A1QGL90H, NVIDIA GPU for CUDA models). They reflect total script runtime including data loading and evaluation, not just `model.fit()`.
- **Random split vs. time-based holdout:** The leaderboard uses random splits for fast iteration. The champion evaluation uses a time-based split because it simulates production: the model trains on historical data and predicts a future month. This is why champion PR-AUC (0.5392) is much lower than leaderboard PR-AUC (0.8887) — random splits leak temporal patterns and inflate all metrics.
- **Threshold policy** and reproducibility checklist are documented in [`artifacts/champion/notes.md`](artifacts/champion/notes.md).
