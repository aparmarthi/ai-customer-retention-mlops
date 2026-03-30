# AI Customer Retention & Decision Intelligence Platform

## Capstone Presentation — Full ML Lifecycle

---

### SLIDE 1: Title

# AI Customer Retention & Decision Intelligence Platform

**From 31 GB of Raw Data to a Production Decision Engine**

Amey | ML Engineering Bootcamp Capstone

Links:
- GitHub: [github.com/aparmarthi/ai-customer-retention-mlops]
- Live Dashboard: [amey-churn-predictor.streamlit.app]

---

### SLIDE 2: The Business Problem

# Every Subscription Business Faces Three Questions

| Question | What This System Delivers |
|---|---|
| **Who will churn?** | Ranked probability score for every subscriber |
| **Who should we target?** | Hybrid policy: budget-bounded top-K or ROI-optimal threshold |
| **What's the financial impact?** | Simulated net ROI under configurable cost assumptions |

**Core principle:** Model probability --> Decision policy --> Financial outcome

This system doesn't just predict churn. It tells you **who to contact**, **how many**, and **what you'll earn** from doing it.

---

### SLIDE 3: The Data

# 31 GB of Real KKBox Subscription Data

| Property | Value |
|---|---:|
| Raw data size | ~31 GB (4 tables) |
| Users | ~1 million subscribers |
| Time range | Jan 2015 -- Feb 2017 |
| Tables | Members, Transactions, User Logs, Labels |
| Processed model table | 193,205 rows, ~40 features |
| Validation churn rate | 1.24% (severe class imbalance) |

**The full 31 GB was processed end-to-end -- not sampled, not approximated.**

---

### SLIDE 4: Data Pipeline Architecture

# 8-Step Reproducible ETL Pipeline

```
Raw CSVs (31 GB)
    |
    v
01: CSV --> Parquet (3.4x compression)     [DuckDB streaming]
02: Build user-month spine                  [DuckDB SQL]
03: Aggregate transactions                  [DuckDB 2-stage]
04: Aggregate user logs                     [DuckDB 2-stage]
05: Join all --> model_table (118 MB)       [pandas]
06: Create sample data for CI              [pandas]
07: Derive recency/tenure/frequency         [DuckDB]
08: Create SageMaker subset                [pandas]
```

**Why DuckDB?** pandas would OOM on 29 GB `user_logs.csv`. DuckDB processes it in streaming mode with 4 GB RAM.

**Key decision:** 2-stage aggregation (daily pre-agg --> user rollup) reduces 400M rows to 118 MB without information loss.

---

### SLIDE 5: Train/Test Split Strategy

# Time-Based Holdout (Not Random Split)

| | Random Split | Time-Based Holdout |
|---|---:|---:|
| ROC-AUC | 0.9875 | **0.9660** |
| PR-AUC | 0.8771 | **0.5392** |
| Churn rate | ~6% | **1.24%** |

**Why time-based?**
- Train on everything up to Jan 31, 2017
- Validate on Feb 2017 (the future the model has never seen)
- This is how it works in production: you train on history and predict tomorrow

Random splits leak temporal patterns and inflate all metrics. The lower time-based numbers are **more honest** and reflect real production performance.

---

### SLIDE 6: Model Benchmarking

# 12 Architectures Benchmarked

| # | Model | PR-AUC | ROC-AUC | Train Time |
|---:|---|---:|---:|---:|
| 1 | **LightGBM** | **0.8887** | 0.9894 | ~3 min |
| 2 | XGBoost | 0.8771 | 0.9875 | ~2 min |
| 3 | CatBoost | 0.8737 | 0.9865 | ~3.7 min |
| 4 | FT-Transformer | 0.8214 | 0.9824 | ~23 min |
| 5 | Random Forest | 0.7935 | 0.9782 | ~5 min |
| 6 | NODE | 0.7719 | 0.9737 | ~11 min |
| 7 | TabNet | 0.5233 | 0.9085 | ~32 min |

*Also evaluated: Logistic Regression, Decision Tree, FLAML AutoML (12 total)*

**Why PR-AUC as primary metric?** With 1.24% churn, ROC-AUC looks great (0.96+) even when the model wastes outreach budget. PR-AUC directly measures how well churners are concentrated at the top of the ranked list.

---

### SLIDE 7: Champion Model

# Champion: LightGBM + FLAML AutoML

**FLAML independently confirmed LightGBM** as the best architecture and found hyperparameters manual tuning missed.

| Metric | Value | Lift |
|---|---:|---:|
| ROC-AUC | **0.9660** | 1.9x vs random |
| PR-AUC | **0.5392** | **43.5x** vs base rate |
| Precision @ top-10K | **18.0%** | 3x vs overall churn |
| Recall @ top-10K | **75.0%** | -- |
| ROI-optimal net ROI | **$17,666** | 1,478 users targeted |

**Key FLAML discoveries:**
- `num_leaves = 1,212` (vs manual 64) -- much deeper trees generalize better
- `reg_alpha = 0.56` -- stronger L1 regularization prevents overfitting
- `learning_rate = 0.036` -- slower learning with 146 iterations

**Model file size: 18 MB** -- trains in 3 minutes, infers in <10ms

---

### SLIDE 8: Explainability (SHAP)

# Top Churn Drivers -- Fully Explainable

*[Insert: reports/shap_summary.png -- SHAP beeswarm plot]*

| Rank | Feature | Business Signal |
|---:|---|---|
| 1 | `auto_renew_rate` | Auto-renewal opt-in -- strongest churn predictor |
| 2 | `cancel_rate` | Historical cancellation ratio -- behavioral fingerprint |
| 3 | `plan_list_price_max` | Peak plan price -- captures price sensitivity |
| 4 | `log_last_date` | Recency of last activity -- disengagement precursor |
| 5 | `membership_expire_date_max` | Subscription horizon -- expiry concentrates risk |

**SHAP confirms the model learned causally plausible signals** -- not spurious correlations. Every prediction is decomposable into per-feature contributions.

---

### SLIDE 9: Decision Policy & ROI

# Two Decision Policies -- Not Just Predictions

### Policy 1: Ops-Friendly Top-K (Primary)

Target the top 10,000 highest-risk subscribers each month.
- Precision: 18% | Recall: 75%
- Fixed budget, no threshold tuning needed

### Policy 2: ROI-Optimal Threshold (Fallback)

Target only subscribers above probability 0.68.
- Precision: 70.6% | Contacts: 1,478
- **Net ROI: $17,666**

*[Insert: reports/threshold_vs_roi.png -- ROI vs threshold curve]*

**ROI Framework:**
```
Net ROI = (TP x save_rate x churn_cost) - (N_targeted x intervention_cost)
```

| Scenario | Cost/Contact | Save Rate | Net Result |
|---|---:|---:|---:|
| Low-cost outreach | $0.50 | 12% | **~$12,200** |
| Incentive offers | $10.00 | 20% | Cost-sensitive |
| ROI-optimal threshold | $5.00 | 20% | **$17,666** |

---

### SLIDE 10: Experiment Tracking (MLflow)

# Full Experiment Governance

All 12 model runs tracked in MLflow experiment `kkbox_churn`:

- **Parameters:** Split policy, feature version, all hyperparameters (auto-flattened)
- **Metrics:** ROC-AUC, PR-AUC, F1, Precision@K, Recall@K
- **Artifacts:** model.pkl, scored validation set, threshold sweep, SHAP plots

**Key design decisions:**
- Atomic artifact bundles -- every metric traces back to the exact data
- Split method audit trail -- records what split *actually ran*, not just what was configured
- Standardized metric naming across all 12 experiments

```bash
mlflow ui  # Compare all 12 runs side-by-side at http://localhost:5000
```

---

### SLIDE 11: Serving Architecture

# Production API: FastAPI + Docker

### Three Endpoints

| Endpoint | Method | Purpose |
|---|---|---|
| `/health` | GET | Model status, artifact paths, feature count |
| `/predict` | POST | Single-record churn probability + threshold label |
| `/predict_batch` | POST | Batch scoring (JSON or CSV upload) + top-K or threshold |

### What Makes It Production-Grade

- **Pydantic validation** -- strict input/output contracts
- **Two serving policies** -- threshold and top-K built into the API
- **Feature alignment** -- missing columns filled NaN, extras dropped (robust to schema drift)
- **Structured logging** -- every request logged with method, path, status, latency
- **Prediction logging** -- probability stats, policy used, inference time per batch
- **16 pytest tests** -- health, single predict, batch CSV, edge cases (all passing)

```bash
# One-command startup
docker compose up --build
# API: http://localhost:8000/docs | Dashboard: http://localhost:8501
```

---

### SLIDE 12: Interactive Dashboard (Streamlit)

# Stakeholder-Facing UI

*[Insert: screenshot of Streamlit app]*

### Three Tabs

| Tab | What It Does |
|---|---|
| **Single Prediction** | Enter one user's features, get churn probability + action label |
| **Batch Scoring** | Upload CSV, score all users, download results with ranks |
| **ROI Simulator** | Adjust cost assumptions with sliders, see real-time ROI impact |

The ROI Simulator lets non-technical stakeholders explore: *"What if we target 5,000 users instead of 10,000? What if intervention costs $2 instead of $5?"*

---

### SLIDE 13: Containerization & CI/CD

# Docker + GitHub Actions Pipeline

### Docker Image (~400 MB)

- `python:3.11-slim` base (not 1.5 GB full image)
- Serving-only dependencies (`requirements-serve.txt`)
- Non-root `appuser` (security baseline)
- `HEALTHCHECK` for orchestrator health detection
- Model artifacts frozen inside the image (atomic deployment)

### CI/CD Pipeline (GitHub Actions)

```
git push main
    |
    v
[1. TEST]  -->  pytest test_api.py (16 tests)
    |                 FAIL? --> blocks deploy
    v
[2. BUILD] -->  Docker build + push to ECR
    |                 Tagged: sha-{commit} + latest
    v
[3. DEPLOY] --> ECS Fargate service update
                     ALB health check: /health
                     Rolling deployment, auto-rollback
```

---

### SLIDE 14: Cloud Training Validation (AWS SageMaker)

# Proving the Pipeline Works in the Cloud

| Aspect | Local | SageMaker |
|---|---|---|
| Purpose | Rapid iteration, 12 experiments | Cloud workflow proof |
| Cost | Free (local hardware) | Pay-per-minute EC2 |
| ROC-AUC | **0.9660** | 0.9484 |
| PR-AUC | **0.5392** | 0.4707 |

**One controlled job demonstrated:**
- S3 data ingestion (train channel)
- Script Mode execution (`train.py`)
- Artifact packaging (`model.tar.gz` --> S3)
- Model Registry integration (versioned, approval-gated)

The local champion remains the production model. SageMaker validates that the same pipeline works in managed cloud infrastructure -- establishing a path for automated retraining.

---

### SLIDE 15: Deployment Architecture

# All-AWS Production Architecture

```
[Data Sources]     [Feature Pipeline]    [Training]        [Registry]
  KKBox DB    -->   DuckDB 8-step   -->  SageMaker    -->  Model Registry
  (31 GB)           ETL (118 MB)         LightGBM          (versioned)
                         |                    |                  |
                         v                    v                  v
                    [S3 Bucket]          [MLflow]          [ECR Image]
                                                               |
                                                               v
[Monitoring]         [Serving]            [CI/CD]         [Artifacts]
 CloudWatch  <--   ECS Fargate   <--   GitHub Actions  <-- model.pkl
 - Latency         (Docker)             1. pytest          threshold.json
 - Errors          FastAPI              2. Docker build    feature_list.json
 - Drift           /predict             3. ECS deploy      metrics.json
                   /predict_batch
 SNS Alerts        /health
                   ALB (load balancer)
```

**Estimated cost: ~$26/month** at low-to-moderate inference volume

---

### SLIDE 16: Monitoring & Model Care

# Deployment Is Just the Beginning

### Three Monitoring Layers

| Layer | What | How | Alert |
|---|---|---|---|
| **Infrastructure** | Latency, errors, memory | CloudWatch + ALB metrics | p95 > 500ms |
| **Model** | Prediction distribution drift | KS-test on monthly batches | Mean shifts > 2 std |
| **Business** | Precision@K, campaign ROI | Actuals join (30-day lag) | ROC-AUC < 0.90 |

### Retraining Triggers

| Trigger | Condition | Response |
|---|---|---|
| Scheduled | Monthly | Full pipeline: pull --> train --> evaluate --> deploy |
| Performance | ROC-AUC < 0.90 | Emergency retrain within 48 hours |
| Drift | >3 features drifting (KS p<0.01) | Investigate, retrain if metrics degrade |

### Rollback Strategy

Every Docker image tagged with git SHA. ECS keeps previous task definitions.
**Rollback time: < 5 minutes.**

---

### SLIDE 17: Scaling Path

# From 1M Users to 1B

| Scale | Users | Raw Data | Compute | Change Required |
|---|---:|---:|---|---|
| **Current** | 1M | 31 GB | DuckDB (local) | None |
| **10x** | 10M | 310 GB | DuckDB (local) | None |
| **100x** | 100M | 3 TB | Spark SQL | Swap engine |
| **1000x** | 1B | 31 TB | BigQuery/Spark | Swap engine |

The 2-stage aggregation pattern, Parquet format, and LightGBM all scale linearly.
**The architecture doesn't require redesign -- only swapping the execution engine.**

---

### SLIDE 18: What This Project Demonstrates

# Full-Stack ML Engineering

| Competency | Evidence |
|---|---|
| **Data Engineering** | 8-step pipeline processing 31 GB with DuckDB; Parquet columnar storage |
| **Model Development** | 12 architectures benchmarked; FLAML AutoML; time-based evaluation |
| **MLOps** | MLflow tracking; SageMaker cloud validation; Model Registry |
| **Evaluation** | PR-AUC under class imbalance; 99-step threshold sweep; ROI simulation |
| **Explainability** | SHAP analysis; business signal mapping; causality-aware interpretation |
| **API Engineering** | FastAPI with Pydantic; two serving policies; 16 pytest tests |
| **Containerization** | Docker slim image; HEALTHCHECK; docker-compose; non-root user |
| **CI/CD** | GitHub Actions: test --> build --> deploy pipeline |
| **Cloud** | AWS SageMaker training + Model Registry; ECS Fargate serving architecture |
| **Business Translation** | ROI framework; cost scenarios; A/B test design |

---

### SLIDE 19: Live Demo

# Try It Yourself

### Live Dashboard (Streamlit)
**[amey-churn-predictor.streamlit.app]**

1. Tab 1: Single prediction with JSON
2. Tab 2: Upload CSV for batch scoring
3. Tab 3: ROI simulator with interactive sliders

### GitHub Repository
**[github.com/aparmarthi/ai-customer-retention-mlops]**

```bash
# Run locally in one command:
docker compose up --build
# API: localhost:8000/docs | Dashboard: localhost:8501
```

---

### SLIDE 20: Key Takeaways

# What I Learned

1. **Deployment is harder than training.** The model was done in Step 4. Steps 5-12 are everything else: policy, explainability, serving, containerization, CI/CD, monitoring, scaling.

2. **The right metric changes everything.** Switching from ROC-AUC to PR-AUC as the primary metric under 1.24% class imbalance completely changed which models looked good and which decisions were defensible.

3. **ML models don't make decisions -- policies do.** A churn probability is useless without a targeting policy. The top-K vs threshold trade-off is a business decision, not a modeling decision.

4. **Scale decisions should be lazy.** DuckDB processes 31 GB on a laptop. Spark would have worked too, but at 10x the setup cost. Choose the simplest tool that works today, and document the migration path for tomorrow.

5. **Write the deployment infrastructure yourself.** Using managed ML platforms hides the complexity. Building FastAPI + Docker + CI/CD from scratch forces you to understand every layer.

---

## Appendix: Links & Resources

| Resource | Link |
|---|---|
| GitHub Repository | [github.com/aparmarthi/ai-customer-retention-mlops] |
| Live Dashboard | [amey-churn-predictor.streamlit.app] |
| Deployment Plan (Step 9) | `reports/step_09_deployment_plan.md` |
| Architecture Doc (Step 10) | `reports/step_10_deployment_architecture.md` |
| Scaling Notebook (Step 8) | `notebooks/08_scaling_prototype.ipynb` |
| Business Assumptions | `reports/business_assumptions.md` |
| SHAP Beeswarm Plot | `reports/shap_summary.png` |
| Threshold vs ROI Plot | `reports/threshold_vs_roi.png` |
| Threshold Sweep Data | `reports/threshold_sweep.csv` |

---

## Image Insertion Guide

When creating the Google Slides, insert these images from the repo:

| Slide | Image | File Path |
|---|---|---|
| Slide 8 (SHAP) | SHAP beeswarm plot | `reports/shap_summary.png` |
| Slide 9 (ROI) | ROI vs threshold curve | `reports/threshold_vs_roi.png` |
| Slide 9 (Threshold) | Precision/Recall vs threshold | `reports/threshold_vs_precision_recall.png` |
| Slide 15 (Architecture) | System architecture diagram | `docs/architecture.png` |
| Slide 12 (Dashboard) | Screenshot of Streamlit app | Take a screenshot while running locally |
