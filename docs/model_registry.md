# Lightweight Model Registry & Lifecycle

This repository uses a **lightweight model registry** pattern to communicate governance and reproducibility without relying on a hosted registry service.

## Terminology

- **Candidate**: a trained model produced during experimentation.
- **Approved model**: a candidate that passes offline quality + ROI checks.
- **Champion**: the current best approved model selected for serving/usage in demos.

## Versioning

Model releases are versioned as `v1`, `v2`, … and are tied to:
- **Git commit SHA**: the exact code used to train/evaluate
- **Data version**: dataset snapshot (build timestamp + fingerprint/hash)
- **Policy**: the decision rule used to convert probabilities into actions
- **Metrics**: offline evaluation summary

## Approval process (simulated)

A candidate is marked “approved” when it passes these gates:

1) **Quality gate**  
   - PR-AUC must not regress beyond an acceptable tolerance  
   - Optional: ROC-AUC stable/improved; calibration acceptable

2) **Business/ROI gate**  
   - Under the documented ROI assumptions, expected ROI is stable or improved  
   - Policy (threshold / top-K) is explicitly recorded

3) **Reproducibility gate**  
   - Code SHA and data version are recorded  
   - Feature list is recorded (or derivable from the pipeline)

This is “simulated” in the sense that the gate is implemented via documentation + reproducible artifacts, not via a formal approval workflow tool.

## Promotion steps (manual, simple)

1) Train candidate model (e.g., LightGBM)
2) Evaluate and run ROI simulation
3) Choose policy (threshold or top-K)
4) Update `reports/champion_metadata.json` with:
   - version number (v1/v2/…)
   - code SHA
   - data version + fingerprint/hash
   - metrics + ROI snapshot
   - policy details
5) Tag the release (optional):
   - `git tag -a model-v1 -m "Champion model v1"`
   - `git push origin --tags`

## Notes

If the project later adopts MLflow Model Registry (or SageMaker Model Registry),
this doc maps directly:
- Candidate → “Staging”
- Approved → “Production candidate”
- Champion → “Production”