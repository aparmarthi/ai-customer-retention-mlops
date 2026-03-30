# Deployment Solution Architecture

## KKBox Customer Churn Prediction — Production Deployment

---

## Architecture Diagram

> **See attached diagram** (created in draw.io / Google Drawings)
>
> Diagram layout reference for creating the visual:
>
> ```
>  [Data Sources]        [Feature Pipeline]       [Training]           [Registry]
>  ┌────────────┐       ┌────────────────┐      ┌─────────────┐     ┌──────────────┐
>  │ KKBox DB   │──────▶│ DuckDB Pipeline│─────▶│ SageMaker   │────▶│ SageMaker    │
>  │ (members,  │       │ 8-step ETL     │      │ Training Job│     │ Model        │
>  │ txns, logs)│       │ 31GB → 118MB   │      │ LightGBM    │     │ Registry     │
>  └────────────┘       │ model_table    │      │ + FLAML     │     │ (versioned,  │
>                       └────────────────┘      └─────────────┘     │  approval-   │
>                              │                       │            │  gated)      │
>                              │                       │            └──────┬───────┘
>                              ▼                       ▼                   │
>                       ┌────────────────┐      ┌─────────────┐           │
>                       │ S3 Bucket      │      │ MLflow       │           │
>                       │ (raw + processed│      │ Experiment  │           │
>                       │  parquet)      │      │ Tracking    │           │
>                       └────────────────┘      └─────────────┘           │
>                                                                         ▼
>  [Monitoring]              [Serving]              [CI/CD]         [Artifacts]
>  ┌──────────────┐    ┌─────────────────┐    ┌──────────────┐   ┌──────────────┐
>  │ CloudWatch   │◀───│ ECS Fargate     │◀───│ GitHub       │◀──│ model.pkl    │
>  │ - Latency    │    │ (Docker)        │    │ Actions      │   │ threshold.json│
>  │ - Error rate │    │                 │    │              │   │ feature_list │
>  │ - Drift      │    │ FastAPI         │    │ 1. pytest    │   │ metrics.json │
>  │              │    │ /predict        │    │ 2. Docker    │   └──────────────┘
>  │ SNS Alerts   │    │ /predict_batch  │    │    build+push│
>  │ → PagerDuty  │    │ /health         │    │ 3. ECS deploy│
>  └──────────────┘    │                 │    └──────────────┘
>                      │ ALB (load       │           │
>                      │  balancer)      │    ┌──────────────┐
>                      └─────────────────┘    │ ECR          │
>                              ▲              │ (container   │
>                              │              │  registry)   │
>                       ┌──────────────┐      └──────────────┘
>                       │ CRM / Ops    │
>                       │ Systems      │
>                       │ (API clients)│
>                       └──────────────┘
> ```

---

## 1. Major Components, Inputs, and Outputs

| Component | Input | Output |
|-----------|-------|--------|
| **Data Pipeline** (DuckDB, 8 scripts) | Raw CSVs: members, transactions, user_logs (31 GB) | `model_table.parquet` (118 MB, 193K rows, ~40 features) |
| **Training** (SageMaker + FLAML) | model_table.parquet | Trained LightGBM model (`model.pkl`, 18 MB) + evaluation metrics |
| **Model Registry** (SageMaker) | model.tar.gz (model + metadata) | Versioned, approval-gated model package |
| **Serving API** (FastAPI on ECS Fargate) | JSON record or CSV batch | Churn probability + action label (threshold or top-K policy) |
| **Monitoring** (CloudWatch + drift detection) | Prediction logs, feature distributions | Alerts, dashboards, retraining triggers |

---

## 2. Data Storage

| Data | Storage | Format | Retention |
|------|---------|--------|-----------|
| Raw source data (31 GB) | S3 (`s3://kkbox-churn/raw/`) | Parquet (3.4x compression from CSV) | Permanent |
| Processed feature tables | S3 (`s3://kkbox-churn/processed/`) | Parquet | Keep last 6 months |
| Trained model artifacts | S3 (via SageMaker) + ECR (in Docker image) | `.pkl` (joblib) inside `model.tar.gz` | Last 6 versions |
| Prediction logs | CloudWatch Logs → S3 (exported) | JSON structured logs | 90 days hot, 1 year cold |
| Experiment metadata | MLflow tracking server | MLflow native | Permanent (audit trail) |

**Decision: S3 as the central data store.** S3 is the natural choice because SageMaker reads/writes directly to S3, and it provides versioning, lifecycle policies, and cross-service access. DuckDB can read from S3 directly for pipeline runs.

---

## 3. Data Flow Between Components

```
Source DB → S3 (raw parquet) → DuckDB pipeline → S3 (model_table)
                                                       │
                    ┌──────────────────────────────────┘
                    ▼
              SageMaker Training Job → S3 (model.tar.gz) → Model Registry
                                                                 │
                    ┌────────────────────────────────────────────┘
                    ▼
              GitHub Actions CI/CD → ECR (Docker image) → ECS Fargate (live API)
                                                                 │
                    ┌────────────────────────────────────────────┘
                    ▼
              CloudWatch (prediction logs) → Drift detection → Retraining trigger
```

All data transfers happen through S3 or container image registries. No direct component-to-component data passing. This decoupled design means any component can be replaced independently.

---

## 4. ML Model Lifecycle

| Phase | What Happens | Tools |
|-------|-------------|-------|
| **Train** | LightGBM trained on model_table with FLAML hyperparameters | SageMaker Training Job |
| **Evaluate** | Score holdout set; compute ROC-AUC, PR-AUC, Precision@K, ROI | Threshold sweep script |
| **Register** | Package model + metadata; submit for approval | SageMaker Model Registry |
| **Approve** | Automated gates (metric thresholds) + manual review | Registry approval status |
| **Deploy** | CI/CD builds new Docker image; deploys to ECS | GitHub Actions + ECR + ECS |
| **Monitor** | Track predictions, drift, latency, business ROI | CloudWatch + custom drift scripts |
| **Retire** | Previous version kept in registry; traffic shifted to new version | ECS blue-green deployment |

---

## 5. Retraining Strategy

**Cadence: Monthly** (aligned with KKBox's subscription billing cycle), plus event-driven triggers.

| Trigger Type | Condition | Response |
|-------------|-----------|----------|
| **Scheduled** | First Monday of each month | Full pipeline: data pull → train → evaluate → deploy |
| **Performance** | ROC-AUC drops below 0.90 on actuals (30-day lag) | Emergency retrain within 48 hours |
| **Data drift** | KS-test flags >3 key features drifting (p < 0.01) | Evaluate; retrain if model metrics degrade |
| **Business** | Subscription pricing or tier changes | Feature engineering update + retrain |

**Retraining data:** Latest 24 months of transaction, membership, and activity data. Stored in S3 as Parquet. The 8-step DuckDB pipeline processes raw data into the feature table. Training uses time-based chronological holdout (most recent month = validation) to prevent temporal leakage.

---

## 6. Retrained Model Evaluation

Before any new model replaces the champion:

| Gate | Pass Condition | Rationale |
|------|---------------|-----------|
| ROC-AUC | >= 0.9560 (current - 0.01) | Ranking quality must not regress |
| PR-AUC | >= 0.5192 (current - 0.02) | Precision under class imbalance (1.24% churn) |
| ROI | >= $15,000 under ROI-optimal policy | Business value must remain positive |
| Smoke test | `/health` returns 200; `/predict` returns valid response on test record | Serving layer integrity |

**Decision: Tolerance bands, not strict improvement.** Requiring strict improvement on every retrain leads to stuck deployments when metrics plateau. Small regressions within tolerance are acceptable if the model is trained on fresher data.

---

## 7. Retrained Model Deployment

1. New model passes evaluation gates → Registry status set to `Approved`
2. GitHub Actions builds new Docker image with updated `artifacts/champion/`
3. Image pushed to ECR with git SHA tag + `latest`
4. ECS service updated with new task definition (rolling deployment)
5. Health check confirms `/health` returns 200 before old tasks drain
6. If health check fails → ECS automatically rolls back to previous task definition

**Rollback time: < 5 minutes** (ECS keeps previous task definition; no rebuild needed).

---

## 8. Model Artifact Storage

| Artifact | Location | Versioning |
|----------|----------|-----------|
| `model.pkl` (18 MB) | S3 via SageMaker + inside Docker image | Model Registry version + git SHA |
| `threshold.json` | Same as model | Versioned together (they're coupled) |
| `feature_list.json` | Same as model | Versioned together |
| `metrics.json` | Same as model + MLflow | Permanent audit trail |
| Docker image | ECR | SHA-tagged (`sha-a1b2c3d`) + `latest` |

**Decision: Model artifacts are committed into the Docker image**, not loaded at runtime from S3. This makes deployments atomic — the exact model + code + config are frozen together. No risk of version mismatch between serving code and model file.

---

## 9. Monitoring and Debugging

| Layer | What | Tool | Alert |
|-------|------|------|-------|
| **Infrastructure** | Latency (p95), error rate, CPU/memory | CloudWatch + ECS metrics | p95 > 500ms, error rate > 1% |
| **Model** | Prediction distribution, label rate drift | Custom script (monthly) | Mean churn probability shifts > 2 std |
| **Data** | Feature distribution drift (KS-test) | Custom script on S3 data | >3 features with p < 0.01 |
| **Business** | Precision@K on actuals, campaign ROI | Actuals join (30-day lag) | ROC-AUC < 0.90 |
| **Debugging** | Request/response logging, SHAP explanations | CloudWatch Logs (structured JSON) | On-demand |

---

## 10. Technology Stack

| Function | Technology | Why This Choice |
|----------|-----------|----------------|
| Data processing | DuckDB + Parquet | Handles 31 GB on single machine; 3.4x compression; SQL interface |
| Model training | LightGBM + FLAML | Champion across 12 architectures; fast, interpretable, small footprint |
| Cloud training | AWS SageMaker (Script Mode) | Managed infrastructure; Model Registry for governance |
| Experiment tracking | MLflow | Open-source; tracks 12+ experiments with artifacts |
| Serving | FastAPI + Uvicorn | Async, auto-docs, Pydantic validation, <10ms inference |
| Containerization | Docker (python:3.11-slim) | ~400 MB image; reproducible; platform-agnostic |
| Container registry | Amazon ECR | Native to AWS; integrates with ECS |
| Compute | AWS ECS Fargate | Serverless containers; no cluster management; auto-scaling |
| CI/CD | GitHub Actions | Native to repo; test → build → deploy pipeline |
| Monitoring | CloudWatch + SNS | Native to AWS; log aggregation, metrics, alerting |
| Data storage | S3 | Central, versioned, cross-service accessible |

---

## 11. Estimated Cost

| Resource | Monthly Cost | Notes |
|----------|-------------|-------|
| ECS Fargate (0.25 vCPU, 0.5 GB, ~10 hrs/day active) | ~$5 | Scale-to-zero with no minimum |
| ECR (image storage) | ~$1 | ~400 MB image, 10 versions |
| S3 (data + artifacts, ~50 GB) | ~$1.15 | Standard tier |
| SageMaker training (1 job/month, ml.m5.large, ~10 min) | ~$0.20 | On-demand, per-second billing |
| CloudWatch (logs + metrics) | ~$3 | 5 GB logs/month |
| ALB (load balancer) | ~$16 | Fixed cost + LCU charges |
| **Total estimated** | **~$26/month** | At low-to-moderate inference volume |

**Cost comparison**: Equivalent SageMaker real-time endpoint (ml.t2.medium, always-on) would cost ~$37/month just for serving — more than the entire ECS architecture above.

---

## 12. Edge Cases, Stress Scenarios, and Resilience (Excellence)

### Handling Edge and Stress Cases

| Scenario | How the Architecture Handles It |
|----------|-------------------------------|
| **Traffic spike** (10x normal batch) | ECS auto-scaling adds Fargate tasks; ALB distributes load; no pre-provisioning needed |
| **Model serves stale predictions** (data pipeline delayed) | Monitoring detects prediction distribution shift; alerts fire; system continues serving last-known-good model |
| **Corrupted model artifact** | Health check at container startup fails → ECS does not route traffic → previous task definition remains active |
| **Feature schema change** (new features added) | `feature_list.json` is versioned with the model; API drops unknown columns and fills missing ones with NaN — graceful degradation |
| **SageMaker training job fails** | Model Registry has no new version → current champion continues serving; alert sent to retrain manually |
| **Cold start latency** | Model loads at container startup (~200ms); ECS can keep minimum 1 task running to eliminate cold starts (adds ~$15/month) |

### Scalability Path

| Scale Level | Architecture | Change Required |
|-------------|-------------|----------------|
| **Current** (~1K predictions/day) | Single ECS task, 0.25 vCPU | None |
| **Medium** (~100K predictions/day) | 2-4 ECS tasks, ALB | Auto-scaling policy only |
| **High** (~1M+ predictions/day) | ECS cluster + SQS for async batches | Add queue; separate batch worker tasks |
| **Enterprise** (multi-model, real-time) | Kubernetes (EKS) + model mesh | Full re-architecture |

### Alternative Architectures by Constraint

| Constraint | Architecture | Trade-off |
|-----------|-------------|-----------|
| **Lowest cost** ($0) | Render.com free tier + model in repo | 30s sleep timeout; 512 MB RAM; no auto-scaling |
| **Simplest ops** (~$5/mo) | AWS Lambda + API Gateway | 250 MB package limit; 15-min timeout; cold starts |
| **Balanced** (~$26/mo) | ECS Fargate + ALB (proposed) | Best cost/control trade-off for single-model serving |
| **Maximum control** (~$70+/mo) | EKS (Kubernetes) | Full orchestration; excessive for single model |
| **Fully managed ML** (~$40+/mo) | SageMaker Endpoints | Built-in A/B testing + Model Monitor; vendor lock-in; always-on cost |

**Decision: ECS Fargate is the proposed architecture** because it provides serverless container management (no EC2 instances to manage), native Docker support (our existing Dockerfile works as-is), auto-scaling, and costs under $30/month — while staying entirely within the AWS ecosystem alongside SageMaker for training.

---

*All infrastructure components described above have working implementations in the repository: Dockerfile, GitHub Actions CI/CD pipeline, FastAPI serving layer, SageMaker training scripts, and model artifacts.*
