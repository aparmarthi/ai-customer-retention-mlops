# Model Registry & Lifecycle

This project uses a **two-tier model registry** approach:

1. **Local (lightweight):** File-based versioning via `artifacts/champion/` + MLflow experiment tracking — used for all development and evaluation
2. **Cloud (SageMaker):** AWS SageMaker Model Registry with versioned, approval-gated Model Packages — demonstrated via a single controlled training job

## Terminology

- **Candidate:** A trained model produced during experimentation
- **Approved:** A candidate that passes offline quality + ROI gates
- **Champion:** The current best approved model selected for serving

## Versioning

Model releases are versioned as `v1`, `v2`, … and are tied to:

| Metadata | Source | Purpose |
|---|---|---|
| Git commit SHA | `git log` | Exact code used to train/evaluate |
| Data version | Dataset snapshot + build timestamp | Reproducible input data |
| Evaluation metrics | `metrics.json` | PR-AUC, ROC-AUC, Precision@K, Recall@K |
| Decision policy | `threshold.json` | Threshold and top-K parameters used for actioning |
| ROI assumptions | `reports/business_assumptions.md` | Cost model linking metrics to business impact |
| Hyperparameters | `flaml_best_params.json` | Full FLAML configuration for reproducibility |

## Approval Process (simulated)

A candidate is marked "approved" when it passes these gates:

### 1. Quality Gate
- PR-AUC must not regress beyond acceptable tolerance
- ROC-AUC stable or improved
- Precision@K at target K does not degrade

### 2. Business / ROI Gate
- Under documented ROI assumptions, expected ROI is stable or improved
- Policy (threshold or top-K) is explicitly recorded

### 3. Reproducibility Gate
- Code SHA and data version are recorded
- Feature list is frozen (`feature_list.json`)
- Split definition is documented (time-based cutoff date)

This is "simulated" in the sense that the gate is implemented via documentation + reproducible artifacts, not via a formal approval workflow tool.

## Promotion Steps

1. Train candidate model
2. Evaluate on time-based holdout and run ROI simulation
3. Choose policy (threshold or top-K)
4. Update `artifacts/champion/` with the new model bundle
5. Log the run in MLflow with all artifacts
6. Optionally register in SageMaker Model Registry (for cloud deployment)
7. Tag the release:
   ```bash
   git tag -a model-v1 -m "Champion model v1 — FLAML LightGBM, PR-AUC 0.5392"
   git push origin --tags
   ```

## How This Maps to Production Registries

| This Project | MLflow Model Registry | SageMaker Model Registry |
|---|---|---|
| `artifacts/champion/` | Model Version (Staging → Production) | Model Package (PendingApproval → Approved) |
| `metrics.json` | Logged metrics per version | Metrics in `model.tar.gz` |
| `threshold.json` | Custom metadata / tags | Companion file in artifact bundle |
| Git tag `model-v1` | Version number | Model Package version |
| Simulated approval | Stage transition (Staging → Production) | `ModelApprovalStatus` (Approved / Rejected) |

SageMaker Model Registry integration is demonstrated in [`cloud/sagemaker/README.md`](../cloud/sagemaker/README.md) — including versioned Model Package creation, approval status management, and artifact lineage.
