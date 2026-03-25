# Phase 5 — Controlled AWS / SageMaker Validation

This folder contains a minimal SageMaker workflow for demonstrating managed cloud training and model registration using the actual KKBox churn champion LightGBM configuration.

## Goal

Show SageMaker literacy with minimal cost by:

1. Uploading the processed training dataset (or a smaller subset) to S3
2. Running exactly one SageMaker training job
3. Saving the model artifact to S3 automatically
4. Registering the trained artifact in SageMaker Model Registry
5. Stopping there — no endpoint deployment

## Why this phase exists

This phase is not about improving the model.

It is a workflow demonstration showing that the project can move from local experimentation into a managed cloud training environment.

### Local experimentation vs cloud validation

| Concern | Local Experimentation | SageMaker Cloud Validation |
|---|---|---|
| **Purpose** | Rapid iteration, model comparison, tuning, debugging | Prove cloud workflow capability |
| **Cost model** | Free (local hardware) | Pay-per-minute EC2 instance |
| **Experiment tracking** | MLflow (local server) | Artifacts auto-packaged to S3 |
| **Iteration speed** | Fast — seconds to restart | Slower — container spin-up, S3 I/O |
| **Number of runs** | 12+ experiments across model families | Single controlled run |
| **Infrastructure** | None required | IAM role, S3 bucket, SageMaker access |

Local experimentation was used for:
- rapid iteration across 12 model architectures (baseline through AutoML)
- cheaper repeated runs during hyperparameter tuning
- model comparison via MLflow leaderboard
- debugging feature engineering and split logic
- SHAP explainability analysis

SageMaker cloud validation was used for:
- demonstrating managed training infrastructure (Script Mode)
- showing S3-based input/output data workflows
- showing model artifact management (auto-packaging to `model.tar.gz`)
- showing Model Registry usage (versioned, approval-gated packages)
- proving AWS / MLOps literacy in a portfolio context

**The local champion remains the production model.** The SageMaker run validates that the same pipeline works in a managed cloud environment — it does not replace local results.

## Why only one SageMaker training job was run

Only one controlled SageMaker job was run because the purpose was to demonstrate cloud workflow capability, not repeat all tuning in the cloud.

Reasons:
- **Lower cost** — SageMaker bills per-second of instance time; running 12+ experiments on `ml.m5.large` would add up with no modeling benefit
- **Faster completion** — one job proves the full workflow end-to-end (submit → train → artifact → registry)
- **Sufficient proof** — a single successful job demonstrates: S3 data ingestion, Script Mode execution, artifact packaging, and Model Registry integration
- **No expected improvement** — the same hyperparameters and split logic are used; cloud training does not change model quality

This is intentionally a minimal validation run. In a production setting, SageMaker would be used for scheduled retraining, hyperparameter tuning jobs, and multi-instance distributed training — but those workflows are cost-prohibitive for a portfolio demonstration.

## Local Champion vs. SageMaker Metrics

| Metric | Local Champion | Lift (Local) | SageMaker Run | Lift (SageMaker) |
|---|---:|---:|---:|---:|
| ROC-AUC | **0.9660** | **1.9x** vs random | 0.9484 | **1.9x** vs random |
| PR-AUC | **0.5392** | **43.5x** vs base rate | 0.4707 | **38.0x** vs base rate |
| F1 | 0.3678 | — | **0.4658** | — |

**Key observations:**

- **ROC-AUC:** Local is +1.8pp higher (0.9660 vs 0.9484) — local model ranks churners better overall
- **PR-AUC:** Local is +6.8pp higher (0.5392 vs 0.4707) — meaningful gap given the low churn rate (1.24%); PR-AUC is the more informative metric under severe class imbalance
- **F1:** SageMaker is +9.8pp higher (0.4658 vs 0.3678) — this is the notable reversal

**Why F1 flips:** The local F1 is `f1_at_0_5` (fixed 0.5 threshold). F1 is sensitive to the decision boundary relative to the score distribution. A model with lower ranking quality can produce a higher F1 if its score distribution places more mass near the 0.5 cutoff. ROC-AUC and PR-AUC are threshold-invariant and remain the more reliable metrics for cross-environment comparison.

**Why metrics differ at all:** The SageMaker run used the same hyperparameters and split logic, but differences in data volume (subset fraction for cost control), row ordering, and the quantile fallback split can shift the validation distribution. With a validation churn rate of only ~1.24%, PR-AUC is especially sensitive to small changes in positive-class composition.

## Files

| File | Runs where | Purpose |
|---|---|---|
| `train.py` | Inside SageMaker container | Reads parquet from `/opt/ml/input/data/train/`, trains LightGBM, writes artifacts to `/opt/ml/model/` |
| `launch_training_job.py` | Local machine | Submits one SageMaker training job using the training script and S3 input data |
| `register_model.py` | Local machine (after training) | Registers the produced `model.tar.gz` artifact in SageMaker Model Registry |
| `requirements.txt` | Inside SageMaker container | Runtime dependencies installed during container setup |

## SageMaker path mental model

Inside the temporary SageMaker training environment:

- input data is made available under `/opt/ml/input/data/<channel_name>/`
- model artifacts must be written under `/opt/ml/model/`

In this project, the channel name is `train`, so the training script reads from:

`/opt/ml/input/data/train/`

The training script writes the final model artifacts to:

`/opt/ml/model/`

SageMaker automatically packages everything in `/opt/ml/model/` into `model.tar.gz` and uploads it to S3.

You do not create these folders locally. SageMaker creates them inside the training container.

## Where artifacts land in S3

### S3 layout

```text
s3://amey-kkbox-sagemaker-us-east-1/kkbox-churn/
    input/
        subset/
            model_table_sagemaker_subset.parquet
    training/
        artifacts/
            kkbox-churn-champion-20260310142800/
                output/
                    model.tar.gz
```

### Actual artifact location

The completed training job produced:

```
s3://amey-kkbox-sagemaker-us-east-1/kkbox-churn/training/artifacts/kkbox-churn-champion-20260310142800/output/model.tar.gz
```

### What is inside `model.tar.gz`

SageMaker auto-packages everything written to `/opt/ml/model/` by `train.py`:

| File | Description |
|---|---|
| `model.pkl` | Trained LightGBM champion model (joblib-serialized) |
| `feature_list.json` | Exact feature columns and order used at train/score time |
| `categorical_cols.json` | Categorical feature list for LightGBM native handling |
| `flaml_best_params.json` | Full FLAML-tuned hyperparameter configuration |
| `metrics.json` | Complete evaluation summary (ROC-AUC, PR-AUC, F1, Precision@K, Recall@K, split metadata) |
| `valid_scored.parquet` | Scored validation set (`msno`, `y_true`, `y_proba`) |

### Downloading and inspecting artifacts

```bash
# Download from S3
aws s3 cp s3://amey-kkbox-sagemaker-us-east-1/kkbox-churn/training/artifacts/kkbox-churn-champion-20260310142800/output/model.tar.gz .

# Extract
tar -xzf model.tar.gz

# Inspect metrics
cat metrics.json
```

## Model Registry versioning

### How it works conceptually

SageMaker Model Registry organizes models into **Model Package Groups** — named collections that hold versioned **Model Packages**. Each package is an immutable snapshot of a trained model artifact plus its inference specification.

```
Model Package Group: "kkbox-churn-champion"
    │
    ├── Model Package v1  (Approved)
    │     ├── model artifact: s3://...model.tar.gz
    │     ├── inference image: sklearn 1.2-1
    │     └── approval status: Approved
    │
    └── Model Package v2  (future re-train)
          └── ...
```

### Registration flow in this project

1. **`register_model.py`** creates the Model Package Group if it does not exist (`kkbox-churn-champion`)
2. It then creates a new **Model Package** within that group, pointing to:
   - The `model.tar.gz` S3 URI from the completed training job
   - The `sklearn 1.2-1` inference container image
3. The package is registered with `ModelApprovalStatus = "Approved"`

### What the approval status means

| Status | Meaning |
|---|---|
| `PendingManualApproval` | Artifact exists but has not been reviewed — cannot be deployed |
| `Approved` | Artifact has passed quality gates — eligible for deployment |
| `Rejected` | Artifact was reviewed and failed quality gates |

In a production pipeline, approval would be gated on automated checks (metric thresholds, data drift tests, A/B test results). In this portfolio project, approval is set directly to demonstrate the registry workflow.

### What metadata travels with each version

Every registered model package is self-describing. The `metrics.json` inside `model.tar.gz` captures:

| Metadata | Purpose |
|---|---|
| `roc_auc`, `pr_auc`, `f1_at_0_5` | Evaluation metrics for approval decisions |
| `precision_at_5pct`, `recall_at_5pct` | Business-relevant top-of-funnel metrics |
| `valid_churn_rate` | Class balance of the validation split — critical for interpreting PR-AUC |
| `n_valid`, `n_train` | Data volume — detects subset vs. full-data runs |
| `best_iteration` | LightGBM early-stopping point — reproducibility |

The companion files (`feature_list.json`, `categorical_cols.json`, `flaml_best_params.json`) lock down the exact feature schema and hyperparameters, so any registered version can be reproduced or audited without access to the original training environment.

### Why this matters for MLOps

- **Versioned lineage:** Every registered model is tied to a specific S3 artifact, training job, and container image — full reproducibility
- **Approval gates:** Models cannot be deployed until explicitly approved — governance built into the workflow
- **Rollback:** If a new model version degrades, you revert to the previous approved version in the registry
- **Decoupled training and deployment:** Training produces artifacts; registry manages which artifact is "live" — separation of concerns
