# AI Customer Retention & Decision Intelligence Platform

![Python](https://img.shields.io/badge/Python-3.11+-blue?logo=python&logoColor=white)
![LightGBM](https://img.shields.io/badge/Champion-LightGBM-brightgreen)
![MLflow](https://img.shields.io/badge/Tracking-MLflow-0194E2?logo=mlflow&logoColor=white)
![FastAPI](https://img.shields.io/badge/Serving-FastAPI-009688?logo=fastapi&logoColor=white)

> End-to-end ML platform built on **~28 GB of real KKBox subscription data** — from raw logs to a production-grade decision policy. Predicts subscriber churn, optimizes intervention targeting, and simulates measurable business ROI.

This project goes beyond model accuracy. It demonstrates full-stack ML system design: scalable feature engineering, experiment governance, two-policy decision engine, and ROI-aligned threshold optimization — mirroring real MLE + MLOps + AI Product workflows.

---

## Results at a Glance

| Metric | Value |
|---|---:|
| ROC-AUC (time-based holdout) | **0.9660** |
| PR-AUC | **0.5392** |
| Precision @ top-10k contacts | **18.0%** — 3x base rate |
| Recall @ top-10k contacts | **75.0%** |
| ROI-optimal policy net ROI | **$17,666** (1,478 users targeted) |
| Outreach policy net savings | **~$12,200** (10,000 users targeted) |

---

## Table of Contents

1. [Business Problem](#business-problem)
2. [System Architecture](#system-architecture)
3. [Dataset](#dataset)
4. [Data Pipeline](#data-pipeline)
5. [Models & Leaderboard](#models--leaderboard)
6. [Champion Model](#champion-model)
7. [Experiment Tracking (MLflow)](#experiment-tracking-mlflow)
8. [Decision Policy Engine](#decision-policy-engine)
9. [ROI Simulation](#roi-simulation)
10. [Inference API (FastAPI)](#inference-api-fastapi)
11. [Explainability & Product Insights](#explainability--product-insights)
12. [Tech Stack](#tech-stack)
13. [Repository Structure](#repository-structure)
14. [Quick Start](#quick-start)
15. [Project Status](#project-status)
16. [What This Demonstrates](#what-this-demonstrates)

---

## Business Problem

Every subscription business faces the same three questions:

| Question | This System's Answer |
|---|---|
| **Who is likely to churn?** | Ranked churn probability score for every subscriber |
| **Who should we target, given budget constraints?** | Hybrid policy: ops-driven top-K or ROI-optimal threshold |
| **What is the expected financial impact?** | Simulated net ROI under configurable cost assumptions |

**Core design principle:** *Model probability → Decision policy → Financial outcome.*

Traditional ML projects optimize ROC-AUC. This system optimizes **business ROI**.

---

## System Architecture

```
Raw Subscription & Transaction Logs  (~28 GB, KKBox)
        |
        v
  ETL & Aggregation  (src/data/  --  7 numbered pipeline scripts)
  |  01_convert_to_parquet   Raw CSV -> columnar format
  |  02_build_spine          User-month observation spine
  |  03_aggregate_txns       Transaction-level behavioral rollups
  |  04_aggregate_logs       User activity log aggregations
  |  05_build_model_table    Join all signals into ML-ready table
  |  06_create_sample        Lightweight dev/test extracts
  |  07_derived_tables       Derived feature tables
        |
        v
  Chronological Train / Validation Split
  |  Train:  txn_last_date <= 2017-01-31
  |  Valid:  Feb 2017 holdout (most recent 20%)
        |
        v
  Model Training  (src/models/  --  12 experiments)
  |  Baseline -> LogReg -> DTree -> XGB -> LGBM -> CatBoost
  |  -> RF -> TabNet -> FT-Transformer -> NODE -> Ensemble -> FLAML AutoML
        |
        v
  MLflow Experiment Tracking + Leaderboard  (leaderboard.md)
        |
        v
  Champion Selection  (artifacts/champion/)
  |  FLAML AutoML -- LightGBM  |  ROC-AUC 0.9660  |  PR-AUC 0.5392
        |
        v
  Threshold Optimization & ROI Sweep  (src/evaluation/)
  |  99-step sweep: 0.01 -> 0.99
  |  Precision, Recall, F1, Precision@K, ROI per threshold
  |  ROI-optimal threshold: t=0.68, ROI=$17,666
  |  Ops-friendly top-10k: Precision=18%, Recall=75%
        |
        v
  Decision Policy Engine  (src/serving/policy.py)
  |  Hybrid:
  |    Primary:   top-K (K=10,000)  -- operational capacity driven
  |    Fallback:  ROI threshold (t=0.68)  -- cost-sensitive contexts
        |
        v
  FastAPI Inference Service  (src/serving/api.py)
        |
        v
  Streamlit Executive Dashboard  [planned]
```

---

## Dataset

**Source:** [KKBox Churn Prediction Challenge (Kaggle)](https://www.kaggle.com/competitions/kkbox-churn-prediction-challenge)

| Property | Value |
|---|---|
| Raw size | ~28 GB |
| Processed model table | ~1M+ rows |
| Validation set | 193,205 rows |
| Churn rate (full dataset) | ~6% |
| Churn rate (time-based valid) | 1.2% |
| Time range | Jan 2015 – Feb 2017 |
| Holdout window | Feb 2017 (chronological) |

Large raw files are excluded from Git. The scored validation set and all champion artifacts are committed for reproducibility.

> **Why the churn rates differ:** The ~6% figure is the overall dataset rate. The Feb 2017 holdout captures a specific month's signal with a different distribution — expected with time-based splits and exactly what makes evaluation realistic. Random splits blend these distributions and inflate all metrics.

---

## Data Pipeline

Seven numbered, independently runnable scripts — no notebook dependencies:

| Script | Purpose |
|---|---|
| `01_convert_to_parquet.py` | Convert raw CSVs to columnar Parquet for fast I/O |
| `02_build_spine.py` | Create user-month observation spine (label join anchor) |
| `03_aggregate_transactions.py` | Transaction-level behavioral rollups per user |
| `04_aggregate_user_logs.py` | Activity log aggregations (listening behavior, etc.) |
| `05_build_model_table.py` | Join all feature tables into the ML-ready model table |
| `06_create_sample_data.py` | Lightweight extracts for dev / CI testing |
| `07_create_derived_tables.py` | Derived feature tables (recency, tenure, frequency) |

**Key engineering decisions:**
- Strict chronological ordering prevents future-signal leakage
- Memory-aware processing for >28 GB raw files
- DuckDB used for large-scale in-process SQL aggregation
- All transformations are scriptable and reproducible

---

## Models & Leaderboard

All models tracked in [`leaderboard.md`](leaderboard.md). Sorted by PR-AUC — the primary metric under class imbalance.

| # | Model | PR-AUC | ROC-AUC |
|---|---|---:|---:|
| 01 | Majority class baseline | — | — |
| 02 | Logistic Regression | — | — |
| 03 | Decision Tree | — | — |
| 04 | XGBoost | 0.8771 | 0.9875 |
| 05 | LightGBM | 0.8887 | 0.9894 |
| 06 | CatBoost | 0.8737 | 0.9865 |
| 07 | Random Forest | 0.7935 | 0.9782 |
| 08 | TabNet | 0.5233 | 0.9085 |
| 09 | FT-Transformer | 0.8214 | 0.9824 |
| 10 | NODE | 0.7719 | 0.9737 |
| 11 | Ensemble (soft vote) | 0.8887 | 0.9894 |
| **12** | **FLAML AutoML — LightGBM** | **Champion** | **Champion** |

> Leaderboard metrics used random splits for comparison speed. **Champion evaluation uses the time-based holdout** — the only number that matters for production.

### Random Split vs. Time-Based Holdout

| | Random Split (XGBoost) | Time-Based Holdout (Champion) |
|---|---:|---:|
| ROC-AUC | 0.9875 | **0.9660** |
| PR-AUC | 0.8771 | **0.5392** |

Lower time-based numbers are more honest — and far more representative of real production performance.

### Class Imbalance Strategy

- Class weighting at training time
- PR-AUC as the primary evaluation metric (not ROC-AUC, not accuracy)
- Precision@K for business-facing evaluation
- Full 99-step threshold sweep to find the ROI-aligned decision boundary

---

## Champion Model

**FLAML AutoML — LightGBM** | MLflow experiment: `kkbox_churn` | Run: `champion_lgbm_time_holdout`

| Metric | Value |
|---|---:|
| ROC-AUC | 0.9660 |
| PR-AUC | **0.5392** |
| F1 @ threshold 0.5 | 0.3678 |
| Precision @ top-5k | 26.9% |
| Precision @ top-10k | **18.0%** |
| Precision @ top-20k | 10.9% |
| Recall @ top-5k | 56.0% |
| Recall @ top-10k | **75.0%** |
| Recall @ top-20k | 90.7% |

**Churn concentration lift:** Top-10k churn rate 18% vs. base rate ~6% — roughly **3x concentration** of churners over random selection.

Artifacts frozen in `artifacts/champion/` — see [`artifacts/champion/notes.md`](artifacts/champion/notes.md) for the reproducibility checklist.

---

## Experiment Tracking (MLflow)

All training runs are tracked in the **`kkbox_churn`** MLflow experiment — params, metrics, and artifacts logged atomically in every run, enabling full reproducibility and side-by-side model comparison across all 12 experiments.

A custom wrapper at `src/utils/mlflow_utils.py` provides safe, robust logging: nested dict flattening, non-numeric filtering, graceful missing-file handling, and standardized metric naming — ensuring consistent, queryable data across every run.

### What Gets Logged Per Run

**Parameters** — logged via `log_params_flat`, which auto-flattens nested dicts using `.` separator:

| Parameter | Champion Value | Purpose |
|---|---|---|
| `cutoff_policy` | `quantile_0.8` | Records the split that *actually ran* — not just the intended config |
| `cutoff_date` | `2016-...` | Exact data boundary applied to this run |
| `feature_version` | `model_table_v1` | Ties each run to a specific data pipeline output |
| `lgbm_params.num_leaves` | `1212` | Nested hyperparams flattened — individually queryable in the UI |
| `lgbm_params.learning_rate` | `0.0358` | All FLAML params captured without manual unpacking |
| `top_k_values` | `[5000, 10000, 20000]` | Business evaluation targets baked into run metadata |

**Metrics** — logged via `log_metrics_safe`, which filters non-numeric values and prevents crashes:

| Metric | Champion Run |
|---|---:|
| `roc_auc` | 0.9660 |
| `pr_auc` | **0.5392** |
| `f1_at_0_5` | 0.3678 |
| `precision_at_5000` | 0.2690 |
| `precision_at_10000` | 0.1801 |
| `precision_at_20000` | 0.1090 |
| `recall_at_5000` | 0.5600 |
| `recall_at_10000` | 0.7498 |
| `recall_at_20000` | 0.9072 |

**Artifacts** — all logged in a single atomic run via `log_artifacts_safe` (skips missing files gracefully):

| Artifact | Description |
|---|---|
| `model.pkl` | Trained champion model |
| `feature_list.json` | Exact feature columns and order used at train/score time |
| `flaml_best_params.json` | Full FLAML hyperparameter configuration |
| `metrics.json` | Complete evaluation summary |
| `valid_scored.parquet` | Scored validation set — the data behind every reported metric |
| `threshold_sweep.csv` | 99-threshold precision, recall, ROI, and Precision@K sweep |
| `threshold.json` | Selected policy parameters (both policies) |
| `threshold_vs_precision_recall.png` | Precision / Recall / F1 vs. threshold plot |
| `threshold_vs_roi.png` | Expected ROI vs. threshold with annotated peak |

### Experiment Structure

```
MLflow Experiment: kkbox_churn
  |
  +-- Run: champion_lgbm_time_holdout
  |     Tags:   stage=champion, model=lgbm
  |     Params: cutoff_policy, feature_version, lgbm_params.*, top_k_values
  |     Metrics: roc_auc, pr_auc, precision_at_*, recall_at_*
  |     Artifacts: model + eval data + all downstream reports
  |
  +-- [all 12 model runs comparable in the same experiment view]
```

### View in the MLflow UI

```bash
mlflow ui
# Open: http://localhost:5000
```

Compare all 12 model runs side-by-side, filter by metric, inspect per-run artifact lineage, and trace every reported number back to the exact data and code that produced it.

### Design Decisions Worth Noting

- **Atomic artifact bundles** — model, `valid_scored.parquet`, and all downstream reports are logged in the same run. Every metric is permanently traceable to the data that generated it — no orphaned checkpoints or mismatched eval files
- **Split method audit trail** — `cutoff_policy` records `quantile_0.8` when the fixed-date fallback triggered, not the intended `fixed_date` config. In production, a model registered with the wrong split metadata can cause silent evaluation-training mismatch bugs months later
- **Graceful phased rollout** — `log_artifacts_safe` skips files that don't exist yet (e.g., SHAP plots), so the training run succeeds and logs everything available without failing on planned-but-not-yet-built outputs
- **Standardized metric naming** — `build_eval_metrics_dict` enforces consistent key names (`precision_at_10000`, not `p@10k`) across all experiments, making cross-run queries reliable

---

## Decision Policy Engine

Located in `src/serving/policy.py`. The production policy is **hybrid**:

```json
{
  "primary_policy":   { "type": "top_k",    "k": 10000 },
  "secondary_policy": { "type": "threshold", "threshold": 0.68 }
}
```

### Policy 1 — Ops-Friendly Top-K *(primary)*

Target the **top-10,000 highest-risk subscribers** by predicted probability, each scoring cycle.

| Metric | Value |
|---|---:|
| Contacts per cycle | 10,000 (fixed) |
| Precision | 18.0% |
| Recall | 74.9% |
| Equiv. threshold | ~0.21 |

**Why top-K is preferred operationally:**
- Contact volume is predictable and budget-bounded every month
- No threshold recalibration needed as score distributions shift over time
- Highest-risk users are always selected
- Simple governance: rank and send

### Policy 2 — ROI-Optimal Threshold *(fallback)*

Used when operational capacity is not the binding constraint — e.g., automated low-cost interventions.

| Metric | Value |
|---|---:|
| Threshold | 0.68 |
| Contacts | 1,478 |
| Precision | 70.6% |
| Recall | 43.5% |
| Estimated net ROI | **$17,666** |

`PolicyDecision` objects include: churn probability, action (`target` / `no_target`), policy used, threshold, rank, and metadata.

---

## ROI Simulation

### Framework

```
Net ROI = (TP x save_rate x churn_cost) - (N_targeted x intervention_cost)

Where:
  TP                = true positives (targeted users who would have churned)
  save_rate         = fraction of targeted churners successfully retained
  churn_cost        = revenue lost per unretained churner
  intervention_cost = cost per outreach contact
```

### Scenario Comparison

| Scenario | Cost/contact | Save rate | Value/save | Net result |
|---|---:|---:|---:|---:|
| **Outreach** (email/SMS/nudge) | $0.50 | 12% | $80 | **~$12,200** |
| **Incentive** (discount/offer) | $10.00 | 20% | $60 | Cost-sensitive |
| **ROI-optimal threshold** | $5.00 | 20% | $120 | **$17,666** |

Full scenario documentation: [`reports/business_assumptions.md`](reports/business_assumptions.md)

### Threshold Sweep Outputs

`src/evaluation/threshold_optimization.py` sweeps 99 thresholds and produces:

| Output | Description |
|---|---|
| `reports/threshold_sweep.csv` | Per-threshold: precision, recall, F1, Precision@K, ROI, n_targeted |
| `artifacts/champion/threshold.json` | Selected policy parameters (both policies) |
| `reports/threshold_vs_precision_recall.png` | Precision / Recall / F1 vs. threshold |
| `reports/threshold_vs_roi.png` | Expected ROI vs. threshold with annotated peak |

---

## Inference API (FastAPI)

Production-ready REST API at `src/serving/api.py` — wraps the champion model and exposes both decision policies over HTTP. Built with FastAPI and Pydantic; all artifacts are loaded at startup from `artifacts/champion/`.

```bash
uvicorn src.serving.api:app --reload
# Swagger UI: http://localhost:8000/docs
```

### Endpoints

| Endpoint | Method | Description |
|---|---|---|
| `/health` | GET | Model load status, artifact paths, feature count, default threshold |
| `/predict` | POST | Single-record churn probability + threshold action label |
| `/predict_batch` | POST | Batch scoring via JSON list or CSV file upload — threshold or top-K policy |

### Two Serving Policies

**Threshold policy** (single record or batch):
```json
POST /predict
{ "record": { ...feature dict... }, "policy": "threshold", "threshold": 0.68 }
```

**Top-K policy** (batch only — ranking requires population context):
```json
POST /predict_batch
{ "records": [...], "policy": "top_k", "k": 10000 }
```

Returns per-record: `churn_probability`, `churn_label`, `rank` (top-K only), `policy_used`, `threshold_used`.

**CSV batch upload** is also supported via multipart form upload to `/predict_batch` — enabling analyst-driven batch scoring without a pipeline dependency.

### Design Notes

- Pydantic input/output models enforce contract at the API boundary — `PredictRequest`, `BatchPredictRequest`, `BatchPredictResponse`
- Feature alignment at inference time: missing columns filled with `NaN`, extra columns silently dropped — robust to schema drift
- Equivalent threshold for the configured top-K is surfaced in the response when it matches `threshold.json`, giving every top-K batch run a threshold anchor for auditability

---

## Explainability & Product Insights

To move beyond black-box prediction, the champion LightGBM model is analyzed with **SHAP** (SHapley Additive exPlanations) — making predictions auditable for engineers and actionable for product teams.

### SHAP Analysis

SHAP decomposes each individual prediction into additive feature contributions:

```
churn_probability  =  base_rate
                    + Σ (per-feature SHAP contributions)
```

Every feature either increases or decreases a user’s churn risk relative to the population average — making each score fully explainable.

**What SHAP enables:**
- Identify the strongest global churn drivers across the model
- Separate risk-increasing signals from protective ones
- Detect when the model is learning noise vs. real behavioral patterns
- Segment users into actionable behavioral cohorts

**Outputs** — computed on a 20,000-row validation sample via `shap.TreeExplainer`:

```bash
python -m src.evaluation.shap_analysis
```

| Output | Description |
|---|---|
| `reports/shap_summary.png` | Beeswarm plot — feature importance + direction across all users |
| `reports/top_features.csv` | Ranked feature list with mean absolute SHAP values |

**Top 5 features by mean |SHAP|** (from the champion run):

| Rank | Feature | Mean \|SHAP\| | Business Signal |
|---:|---|---:|---|
| 1 | `auto_renew_rate` | 0.593 | Auto-renewal opt-in rate — the single strongest churn predictor; users not opting in are signaling exit intent |
| 2 | `cancel_rate` | 0.499 | Historical cancellation ratio — past behavior is a very strong fingerprint for future churn |
| 3 | `plan_list_price_max` | 0.329 | Peak plan price seen — captures price sensitivity and premium-tier exposure |
| 4 | `log_last_date` | 0.281 | Recency of last activity — disengagement precursor; a leading indicator before cancellation |
| 5 | `membership_expire_date_max` | 0.247 | Subscription horizon — proximity to expiry concentrates churn risk |

### Model-Derived Risk Signals

SHAP confirms that the model has learned interpretable, causally-plausible churn signals — not spurious correlations:

| Risk Signal | Key Feature | Business Interpretation |
|---|---|---|
| Auto-renewal opt-out pattern | `auto_renew_rate` | Strongest signal — low auto-renewal rate directly reflects disengagement or intent to cancel |
| Historical cancellation behavior | `cancel_rate` | Past cancellations strongly predict future churn; the model learned this behavioral fingerprint |
| Declining recent activity | `log_last_date` | Recency of last activity is a leading indicator — disengagement precedes cancellation |
| Imminent subscription expiry | `membership_expire_date_max` | Expiry proximity concentrates churn risk — intervention window is narrow |
| Short tenure | `txn_tenure_days_approx` | Early lifecycle churn — product fit or onboarding issue |

These signals align with real-world subscription churn behavior and provide a mechanism to **validate the model’s internal logic** beyond held-out accuracy.

### Translating Risk Into Action

Model scores and SHAP cohort signals together enable targeted retention strategies:

| Intervention | Channel | When to Use |
|---|---|---|
| Proactive outreach | Email / SMS / in-app push | Default for all top-K users |
| Renewal reminder | Email / push notification | Users within 7–14 days of expiry |
| Payment resolution | Support + email | Users with recent payment failures |
| Friction reduction | UX / product flow | Users dropping off at renewal step |
| Human-assisted retention | Support team | High-LTV users in top decile of risk |

### Experimental Validation *(AI Product Lens)*

Churn models predict *who will churn* — not *who will respond to intervention*. Without a control group, measured ROI conflates natural churn with model-driven saves. The correct validation approach:

1. Score all subscribers monthly
2. Select the top-K highest-risk users
3. Randomly split into: **50% Treatment** (retention campaign) / **50% Control** (no contact)
4. Measure churn rate difference between groups after 30 days

This isolates true incremental lift:

```
True incremental ROI  =  (churn_rate_control − churn_rate_treatment)
                       × N_treatment
                       × LTV
                       − intervention_cost
```

This is the correct metric — it separates predictive correlation from **causal retention uplift**.

---

## Tech Stack

| Category | Libraries |
|---|---|
| **Data & compute** | NumPy, Pandas, PyArrow, DuckDB |
| **ML — gradient boosting** | LightGBM, XGBoost, CatBoost |
| **ML — deep learning** | PyTorch, pytorch-tabnet (TabNet, FT-Transformer, NODE) |
| **AutoML** | FLAML |
| **Experiment tracking** | MLflow |
| **Explainability** | SHAP |
| **Serving** | FastAPI, uvicorn |
| **Dashboard** | Streamlit *(planned)* |
| **Visualization** | Matplotlib, Seaborn |
| **Utilities** | scikit-learn, joblib, tqdm, fsspec |

---

## Repository Structure

```
ai-customer-retention-mlops/
|
|-- data/                           # Raw, processed, sample data  (large files gitignored)
|-- notebooks/                      # Exploratory analysis
|-- reports/                        # Business assumptions, threshold sweep CSV, plots
|-- scripts/
|   `-- generate_leaderboard.py
|-- docs/
|   `-- KKBox Churn Prediction Capstone Project Proposal.pdf
|-- artifacts/
|   `-- champion/                   # Frozen model bundle
|       |-- model.pkl
|       |-- threshold.json          # Decision policy (ROI-optimal + ops top-K)
|       |-- deployment_policy.json  # Hybrid policy definition
|       |-- metrics.json
|       |-- feature_list.json
|       |-- flaml_best_params.json
|       |-- roi_policy.json
|       |-- valid_scored.parquet    # Scored validation set (msno, y_true, y_proba)
|       `-- notes.md
|
|-- src/
|   |-- data/                       # ETL pipeline  (01 --> 07 numbered scripts)
|   |-- models/                     # 01_baseline --> 12_automl_flaml + 13/14/15 champion scripts
|   |-- evaluation/
|   |   |-- threshold_optimization.py
|   |   `-- shap_analysis.py        # SHAP TreeExplainer -- beeswarm + feature ranking
|   |-- serving/
|   |   |-- api.py                  # FastAPI inference service  (GET /health, POST /predict, POST /predict_batch)
|   |   |-- policy.py               # PolicyDecision engine  (top-K + threshold)
|   |   `-- test_policy.py
|   |-- pipelines/
|   |   `-- run_pipeline.py         # End-to-end orchestrator  [planned]
|   |-- deployment/
|   |   |-- app.py                  # [planned]
|   |   |-- ltv_roi.py              # LTV / ROI serving helper  [planned]
|   |   `-- Dockerfile              # [planned]
|   |-- ui/
|   |   `-- streamlit_app.py        # Executive analytics dashboard  [planned]
|   `-- utils/
|       |-- config.py
|       `-- run_logger.py
|
|-- leaderboard.md
|-- requirements.txt
|-- requirements-dev.txt
|-- docker_compose.yml
`-- README.md
```

---

## Quick Start

### Install

```bash
pip install -r requirements.txt
```

### Run the data pipeline

```bash
python src/data/01_convert_to_parquet.py
python src/data/02_build_spine.py
python src/data/03_aggregate_transactions.py
python src/data/04_aggregate_user_logs.py
python src/data/05_build_model_table.py
```

### Train the champion model (with MLflow tracking)

```bash
python -m src.models.13_train_champion_lgbm_mlflow
```

Logs params, metrics, and all artifacts to the `kkbox_churn` MLflow experiment automatically.

### View experiments in MLflow UI

```bash
mlflow ui
# Open: http://localhost:5000
```

### Score validation set

```bash
python src/models/14_score_valid_champion.py
```

### Run threshold sweep & generate plots

```bash
python src/evaluation/threshold_optimization.py
```

### Run SHAP explainability analysis

```bash
python -m src.evaluation.shap_analysis
# Outputs: reports/shap_summary.png, reports/top_features.csv
```

### Launch the FastAPI inference service

```bash
uvicorn src.serving.api:app --reload
# Swagger UI: http://localhost:8000/docs
# Endpoints: GET /health  |  POST /predict  |  POST /predict_batch
```

### Generate leaderboard

```bash
python scripts/generate_leaderboard.py
```

### Use the policy engine directly

```python
from src.serving.policy import apply_threshold, apply_topk_to_batch

# Single-user threshold policy
decision = apply_threshold(prob=0.82, threshold=0.68)
# PolicyDecision(action='target', policy_used='threshold', ...)

# Batch top-K policy
ranks = apply_topk_to_batch(probs=score_array, k=10_000)
```

---

## Project Status

| Component | Status |
|---|---|
| Data pipeline (ETL, feature engineering) | Complete |
| Model training — 12 experiments | Complete |
| MLflow tracking + leaderboard | Complete |
| Champion model selection | Complete |
| Threshold optimization & ROI sweep | Complete |
| Decision policy engine | Complete |
| FastAPI inference service | Complete |
| SHAP explainability layer | Complete |
| Streamlit executive dashboard | Planned |
| Dockerfile / docker-compose | Planned |
| End-to-end pipeline orchestrator | Planned |

---

## What This Demonstrates

### For Machine Learning Engineer Roles

- End-to-end ML pipeline: raw data through scored predictions, decision policy, and a live serving API
- Principled handling of severe class imbalance at million-row scale
- Time-based cross-validation that reflects production realities — and documents *why* random splits mislead
- AutoML + manual model comparison with a governed, reproducible leaderboard
- Threshold optimization tied directly to a business cost function, not ML metrics
- Dual-policy decision engine with documented trade-offs between coverage and efficiency
- Production REST API (FastAPI + Pydantic) exposing both policies with full request/response contracts
- SHAP explainability layer producing auditable feature contributions from a tree ensemble

### For MLOps / Data Engineering Roles

- Reproducible, script-driven data pipeline with clear stage separation — no notebook dependencies
- Custom MLflow wrapper (`src/utils/mlflow_utils.py`) with nested param flattening, safe metric logging, and graceful artifact handling — not just a `mlflow.log_metric` call
- Atomic run logging: model + scored validation data + downstream reports in one run, making every metric permanently traceable
- Split method audit trail: `cutoff_policy` records what actually ran, not what was configured — production-grade reproducibility thinking
- SHAP explainability pipeline (`src/evaluation/shap_analysis.py`) producing beeswarm plots and ranked feature CSV — model auditing built in, not bolted on
- FastAPI serving layer with two policy modes, Pydantic contract enforcement, and CSV batch upload — separation of serving logic from model training
- Modular `src/` layout designed for CI/CD integration
- Numbered ETL scripts (01→07) for explicit dependency ordering
- DuckDB for large-scale in-process SQL aggregation on raw files

### For Business / Analytics / Product Roles

- ROI simulation framework connecting model Precision@K to net financial impact
- Two defensible policies with documented trade-offs: coverage vs. per-contact efficiency
- Budget-bounded top-K targeting for predictable monthly operational load
- Scenario modeling across intervention types: outreach vs. incentive offers
- Business assumptions documented separately — not baked silently into model training
- Executive dashboard for threshold and ROI scenario exploration *(planned)*

---

*Built on the [KKBox Churn Prediction](https://www.kaggle.com/competitions/kkbox-churn-prediction-challenge) dataset.*
