# Mini-Project: End-to-End Churn Prediction Using SageMaker

**Author:** Amey Parmarthi
**Repository:** [github.com/aparmarthi/ai-customer-retention-mlops](https://github.com/aparmarthi/ai-customer-retention-mlops)
**SageMaker Code:** [`cloud/sagemaker/`](https://github.com/aparmarthi/ai-customer-retention-mlops/tree/main/cloud/sagemaker)

---

## 1. Project Overview

This document consolidates all AWS SageMaker work performed as part of the AI Customer Retention & Decision Intelligence Platform capstone project. The goal was to demonstrate an end-to-end cloud ML workflow: data preparation, model training, evaluation, artifact management, and model registration — all using Amazon SageMaker.

### Problem Statement

Predict subscriber churn for KKBox (a music streaming service) using ~31 GB of real subscription data (~1 million users). The champion model is a FLAML-tuned LightGBM classifier evaluated under severe class imbalance (1.24% churn rate in the time-based holdout).

### Why SageMaker

SageMaker was used to validate that the local training pipeline translates to managed cloud infrastructure without code changes. This demonstrates AWS/MLOps literacy and establishes a path for automated retraining at scale.

| Concern | Local Experimentation | SageMaker Cloud Validation |
|---|---|---|
| **Purpose** | Rapid iteration, model comparison, tuning | Prove cloud workflow capability |
| **Cost model** | Free (local hardware) | Pay-per-minute EC2 instance |
| **Experiment tracking** | MLflow (local server) | Artifacts auto-packaged to S3 |
| **Number of runs** | 12+ experiments across model families | Single controlled run |
| **Infrastructure** | None required | IAM role, S3 bucket, SageMaker access |

---

## 2. Architecture & Workflow

### End-to-End SageMaker Pipeline

```
[Local Machine]                       [AWS Cloud]

08_data_subset_for_sagemaker.py       S3 Bucket
  Creates 10% stratified subset  -->  s3://amey-kkbox-sagemaker-us-east-1/
  (model_table_sagemaker_subset.parquet)    kkbox-churn/input/subset/

launch_training_job.py                SageMaker Training Job
  Submits training job via SDK   -->  - Spins up ml.m5.large instance
  Passes hyperparameters              - Installs lightgbm from requirements.txt
  Points to S3 input data            - Executes train.py (Script Mode)
                                      - Reads from /opt/ml/input/data/train/
                                      - Writes artifacts to /opt/ml/model/
                                      - Auto-packages to model.tar.gz --> S3

register_model.py                     SageMaker Model Registry
  Registers trained artifact     -->  - Creates Model Package Group
  in Model Registry                   - Registers versioned Model Package
                                      - Sets approval status: Approved
```

### SageMaker Training Job Configuration

| Property | Value |
|---|---|
| **Job type** | SageMaker Training Job (Script Mode) |
| **Training image** | `sklearn 1.2-1` (pre-built SageMaker container) |
| **Instance type** | `ml.m5.large` |
| **Entry script** | `cloud/sagemaker/train.py` |
| **Dependencies** | `cloud/sagemaker/requirements.txt` (lightgbm>=4.3) |
| **Input channel** | `train` — S3 parquet via `/opt/ml/input/data/train/` |
| **Output path** | `s3://amey-kkbox-sagemaker-us-east-1/kkbox-churn/training/artifacts/` |
| **Job name** | `kkbox-churn-champion-20260310142800` |
| **Hyperparameters** | Identical to local champion (FLAML-tuned LightGBM) |

### S3 Artifact Layout

```
s3://amey-kkbox-sagemaker-us-east-1/kkbox-churn/
    input/
        subset/
            model_table_sagemaker_subset.parquet     (13 MB, 10% stratified subset)
    training/
        artifacts/
            kkbox-churn-champion-20260310142800/
                output/
                    model.tar.gz                      (auto-packaged by SageMaker)
```

### Contents of `model.tar.gz`

| File | Description |
|---|---|
| `model.pkl` | Trained LightGBM champion model (joblib-serialized) |
| `feature_list.json` | Exact feature columns and order used at train/score time |
| `categorical_cols.json` | Categorical feature list for LightGBM native handling |
| `flaml_best_params.json` | Full FLAML-tuned hyperparameter configuration |
| `metrics.json` | Complete evaluation summary (ROC-AUC, PR-AUC, F1, Precision@K, Recall@K) |
| `valid_scored.parquet` | Scored validation set (msno, y_true, y_proba) |

---

## 3. Data Preparation

### Source Data

- **Original dataset:** ~31 GB of KKBox subscription data (4 tables: Members, Transactions, User Logs, Labels)
- **Processed feature table:** 193,205 rows x ~40 features (`model_table.parquet`, 118 MB)
- **SageMaker subset:** 10% stratified sample = ~19,320 rows (`model_table_sagemaker_subset.parquet`, 13 MB)

### Subsetting Script (`src/data/08_data_subset_for_sagemaker.py`)

```python
import pandas as pd

df = pd.read_parquet("data/kkbox/processed/model_table.parquet")
df_subset = df.sample(frac=0.10, random_state=42)
df_subset.to_parquet(
    "data/kkbox/processed/model_table_sagemaker_subset.parquet", index=False
)
```

**Why subset?** Cost control — running 12+ experiments on `ml.m5.large` with the full dataset would add significant cloud cost with no modeling benefit. The subset preserves class balance via stratified sampling.

### Upload to S3

```bash
aws s3 cp data/kkbox/processed/model_table_sagemaker_subset.parquet \
    s3://amey-kkbox-sagemaker-us-east-1/kkbox-churn/input/subset/
```

---

## 4. Training Script (`cloud/sagemaker/train.py`)

The training script runs inside the SageMaker container and follows the SageMaker Script Mode contract:

### Key Design Decisions

1. **SageMaker path convention:**
   - Input: reads parquet from `/opt/ml/input/data/train/` (SageMaker-provided)
   - Output: writes all artifacts to `/opt/ml/model/` (SageMaker auto-packages to `model.tar.gz`)

2. **Chronological train/valid split:**
   - Primary: fixed date cutoff at `2017-01-31` (same as local champion)
   - Fallback: 80th percentile quantile split if no rows exist after cutoff
   - This mirrors production conditions where you train on history and predict the future

3. **FLAML-tuned LightGBM hyperparameters** (identical to local champion):

```python
LGBM_PARAMS = {
    "colsample_bytree": 0.784575377162775,
    "learning_rate": 0.03583753342568752,
    "max_bin": 1023,
    "min_child_samples": 28,
    "n_estimators": 146,
    "num_leaves": 1212,
    "reg_alpha": 0.5616512686484578,
    "reg_lambda": 0.0009765625,
    "n_jobs": -1,
    "verbose": -1,
    "random_state": 42,
}
```

4. **Feature engineering inside the container:**
   - Drops datetime columns, ID column, and target
   - Converts object columns to `category` dtype for LightGBM native handling
   - Handles KKBox integer date format (e.g., `20170228`) via explicit `%Y%m%d` parsing

5. **Evaluation metrics computed:**
   - ROC-AUC, PR-AUC, F1@0.5 threshold
   - Precision@K and Recall@K for K = 5,000 / 10,000 / 20,000

6. **Artifacts saved to `/opt/ml/model/`:**
   - `model.pkl`, `feature_list.json`, `categorical_cols.json`, `flaml_best_params.json`, `metrics.json`, `valid_scored.parquet`

---

## 5. Job Launcher (`cloud/sagemaker/launch_training_job.py`)

Uses **SageMaker SDK v2** (new core API) to submit the training job:

```python
from sagemaker.train import ModelTrainer
from sagemaker.core.training.configs import Compute, InputData, OutputDataConfig, SourceCode

trainer = ModelTrainer(
    training_image=training_image,       # sklearn 1.2-1
    source_code=SourceCode(
        source_dir=_SCRIPT_DIR,
        entry_script="train.py",
        requirements="requirements.txt",
    ),
    role=args.role_arn,
    compute=Compute(instance_type="ml.m5.large", instance_count=1),
    output_data_config=OutputDataConfig(s3_output_path=output_path),
    hyperparameters={
        "target-col": "is_churn",
        "time-col": "txn_last_date",
        "id-col": "msno",
        "subset-fraction": "1.0",
        "random-state": "42",
    },
)

trainer.train(input_data_config=[InputData(channel_name="train", data_source=train_s3_uri)])
```

### CLI Usage

```bash
python cloud/sagemaker/launch_training_job.py \
    --region us-east-1 \
    --role-arn arn:aws:iam:::<ACCOUNT_ID>:role/SageMakerExecutionRole \
    --bucket amey-kkbox-sagemaker-us-east-1 \
    --train-s3-uri s3://amey-kkbox-sagemaker-us-east-1/kkbox-churn/input/subset/ \
    --instance-type ml.m5.large \
    --wait
```

---

## 6. Model Registration (`cloud/sagemaker/register_model.py`)

After training completes, the model artifact is registered in **SageMaker Model Registry**:

```python
sm_client.create_model_package(
    ModelPackageGroupName="kkbox-churn-champion",
    ModelApprovalStatus="Approved",
    InferenceSpecification={
        "Containers": [{
            "Image": inference_image_uri,          # sklearn 1.2-1 inference
            "ModelDataUrl": model_artifact_s3_uri,  # model.tar.gz in S3
        }],
        "SupportedContentTypes": ["application/json"],
        "SupportedResponseMIMETypes": ["application/json"],
    },
)
```

### Model Registry Structure

```
Model Package Group: "kkbox-churn-champion"
    |
    +-- Model Package v1  (Approved)
          +-- model artifact: s3://...model.tar.gz
          +-- inference image: sklearn 1.2-1
          +-- approval status: Approved
```

### Approval Workflow

| Status | Meaning |
|---|---|
| `PendingManualApproval` | Artifact exists but not reviewed — cannot be deployed |
| `Approved` | Passed quality gates — eligible for deployment |
| `Rejected` | Failed quality gates |

In production, approval would be gated on automated metric thresholds, drift tests, and A/B test results. For this project, approval was set directly to demonstrate the registry workflow.

### CLI Usage

```bash
python cloud/sagemaker/register_model.py \
    --region us-east-1 \
    --model-package-group-name kkbox-churn-champion \
    --model-artifact-s3-uri s3://amey-kkbox-sagemaker-us-east-1/kkbox-churn/training/artifacts/kkbox-churn-champion-20260310142800/output/model.tar.gz \
    --approval-status Approved
```

---

## 7. Results & Evaluation

### Local Champion vs. SageMaker Run

| Metric | Local Champion | SageMaker Run | Notes |
|---|---:|---:|---|
| **ROC-AUC** | **0.9660** | 0.9484 | Local +1.8pp higher |
| **PR-AUC** | **0.5392** | 0.4707 | Local +6.8pp higher (primary metric) |
| **F1@0.5** | 0.3678 | **0.4658** | SageMaker higher due to threshold sensitivity |
| **Lift vs random (ROC)** | 1.9x | 1.9x | Both strong |
| **Lift vs base rate (PR)** | 43.5x | 38.0x | Both demonstrate massive lift |

### Why Metrics Differ

The SageMaker run used identical hyperparameters and split logic, but differences arise from:

1. **Data volume:** 10% stratified subset vs. full dataset
2. **Row ordering:** Different physical ordering after subsetting affects LightGBM leaf assignment
3. **Quantile fallback split:** The subset may trigger the quantile-based split fallback, shifting validation distribution
4. **PR-AUC sensitivity:** With only 1.24% churn, PR-AUC is especially sensitive to small changes in positive-class composition

### Why F1 Reverses

F1 is threshold-dependent (fixed at 0.5). A model with lower ranking quality can produce a higher F1 if its score distribution places more mass near the 0.5 cutoff. **ROC-AUC and PR-AUC are threshold-invariant and remain the more reliable metrics** for cross-environment comparison.

### Key Takeaway

The local champion remains the production model. The SageMaker run validates that the same pipeline works in managed cloud infrastructure — establishing a path for automated retraining.

---

## 8. MLOps Capabilities Demonstrated

| Capability | How It Was Demonstrated |
|---|---|
| **Script Mode Training** | Custom `train.py` with `requirements.txt`, executed inside SageMaker container |
| **S3 Data Workflows** | Input data uploaded to S3, model artifacts auto-packaged and stored in S3 |
| **Hyperparameter Passing** | CLI-style hyperparameters passed through SageMaker SDK to training script |
| **Model Artifact Management** | SageMaker auto-packages `/opt/ml/model/` into `model.tar.gz` and uploads to S3 |
| **Model Registry** | Versioned, approval-gated Model Packages in SageMaker Model Registry |
| **Cost-Controlled Validation** | Stratified subsetting + single controlled job for minimal cloud spend |
| **Reproducibility** | Same FLAML-tuned hyperparameters, split logic, and feature engineering as local |
| **Two-Tier Registry** | Local (artifacts/champion/ + MLflow) + Cloud (SageMaker Model Registry) |

---

## 9. File Inventory

| File | Location | Purpose |
|---|---|---|
| `train.py` | `cloud/sagemaker/` | Training script (runs inside SageMaker container) |
| `launch_training_job.py` | `cloud/sagemaker/` | Submits SageMaker training job from local machine |
| `register_model.py` | `cloud/sagemaker/` | Registers trained model in SageMaker Model Registry |
| `requirements.txt` | `cloud/sagemaker/` | Runtime dependencies for SageMaker container |
| `README.md` | `cloud/sagemaker/` | Detailed documentation of the SageMaker workflow |
| `08_data_subset_for_sagemaker.py` | `src/data/` | Creates 10% stratified subset for cost control |
| `model_registry.md` | `docs/` | Two-tier registry design and approval process |
| `model_table_sagemaker_subset.parquet` | `data/kkbox/processed/` | 13 MB training data subset |

---

## 10. How to Reproduce

### Prerequisites

- AWS account with SageMaker access
- IAM role with SageMaker execution permissions
- S3 bucket for data and artifacts
- Python 3.11+ with `sagemaker`, `boto3` installed

### Steps

```bash
# 1. Create the data subset (from project root)
python src/data/08_data_subset_for_sagemaker.py

# 2. Upload to S3
aws s3 cp data/kkbox/processed/model_table_sagemaker_subset.parquet \
    s3://<YOUR_BUCKET>/kkbox-churn/input/subset/

# 3. Launch training job
python cloud/sagemaker/launch_training_job.py \
    --region us-east-1 \
    --role-arn arn:aws:iam::<ACCOUNT_ID>:role/SageMakerExecutionRole \
    --bucket <YOUR_BUCKET> \
    --train-s3-uri s3://<YOUR_BUCKET>/kkbox-churn/input/subset/ \
    --wait

# 4. Register model in Model Registry
python cloud/sagemaker/register_model.py \
    --region us-east-1 \
    --model-package-group-name kkbox-churn-champion \
    --model-artifact-s3-uri s3://<YOUR_BUCKET>/kkbox-churn/training/artifacts/<JOB_NAME>/output/model.tar.gz

# 5. (Optional) Download and inspect artifacts
aws s3 cp s3://<YOUR_BUCKET>/kkbox-churn/training/artifacts/<JOB_NAME>/output/model.tar.gz .
tar -xzf model.tar.gz
cat metrics.json
```

---

## 11. Production Scaling Path

While this project used a single controlled job for cost efficiency, SageMaker enables:

| Production Use Case | SageMaker Feature |
|---|---|
| **Scheduled retraining** | SageMaker Pipelines with scheduled triggers |
| **Hyperparameter tuning** | SageMaker Automatic Model Tuning (Bayesian optimization) |
| **Distributed training** | Multi-instance training jobs with data parallelism |
| **Real-time inference** | SageMaker Endpoints with auto-scaling |
| **Batch inference** | SageMaker Batch Transform for monthly scoring |
| **Feature store** | SageMaker Feature Store for online/offline feature serving |
| **Model monitoring** | SageMaker Model Monitor for data/model drift detection |
| **CI/CD** | SageMaker Pipelines + CodePipeline for automated ML workflows |
