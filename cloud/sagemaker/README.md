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

Local experimentation was used for:
- rapid iteration
- cheaper repeated runs
- model comparison and tuning
- debugging
- MLflow-based experiment tracking

SageMaker cloud validation was used for:
- demonstrating managed training infrastructure
- showing S3-based input/output workflows
- showing model artifact management
- showing Model Registry usage
- proving AWS / MLOps awareness

## Why only one SageMaker training job was run

Only one controlled SageMaker job was run because the purpose was to demonstrate cloud workflow capability, not repeat all tuning in the cloud.

Reasons:
- lower cost
- faster completion
- enough to prove training-job literacy
- enough to prove artifact handling and model registration

This is intentionally a minimal validation run.

## Does this change model metrics?

Conceptually, no.

This phase uses the same champion modeling logic and the same tuned LightGBM parameters. It is a workflow demonstration, not a new modeling method.

If a subset is used for cost control, metrics may differ slightly due to fewer rows, but the purpose remains cloud workflow validation rather than performance improvement.

## Files

- `train.py`  
  Runs inside SageMaker. Reads parquet data from the SageMaker input channel and saves champion artifacts to the SageMaker model directory.

- `launch_training_job.py`  
  Runs on the local machine. Submits one SageMaker training job using the local training script and S3 input data.

- `register_model.py`  
  Runs on the local machine after training completes. Registers the produced `model.tar.gz` artifact in SageMaker Model Registry.

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

Suggested S3 layout:

```text
s3://<bucket>/kkbox-churn/
    input/
        subset/
            model_table_sagemaker_subset.parquet
    training/
        artifacts/
            <training-job-name>/
                output/
                    model.tar.gz