# KKBox Churn — Model Leaderboard

Sorted by **PR-AUC (Average Precision)**, then ROC-AUC. Metrics are on the **valid split**.

| Rank | Model | Run ID | PR-AUC | ROC-AUC | F1 | Precision | Recall | Accuracy | Threshold |
|---:|---|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | lightgbm | lightgbm-0001 | 0.8887 | 0.9894 | 0.7845 | 0.6805 | 0.9260 | 0.9675 | 0.5000 |
| 2 | ensemble-softvote | ensemble-softvote-0001 | 0.8887 | 0.9894 | 0.7845 | 0.6805 | 0.9260 | 0.9675 | 0.5000 |
| 3 | xgboost-trainapi | xgboost-trainapi-0001 | 0.8771 | 0.9875 | — | — | — | — | 0.5000 |
| 4 | catboost | catboost-0001 | 0.8737 | 0.9865 | 0.7282 | 0.6002 | 0.9257 | 0.9558 | 0.5000 |
| 5 | ft-transformer | ft-transformer-0001 | 0.8214 | 0.9824 | 0.6825 | 0.5374 | 0.9350 | 0.9444 | 0.5000 |
| 6 | random-forest | random-forest-0001 | 0.7935 | 0.9782 | 0.5798 | 0.4188 | 0.9419 | 0.9127 | 0.5000 |
| 7 | node-oblivious | node-oblivious-0001 | 0.7719 | 0.9737 | 0.5334 | 0.3717 | 0.9446 | 0.8944 | 0.5000 |
| 8 | tabnet | tabnet-0005 | 0.5233 | 0.9085 | 0.3998 | 0.5671 | 0.3087 | 0.9407 | 0.5000 |



## Notes

- PR-AUC is the primary metric because churn is highly imbalanced (~6.4% positive).
- Threshold-dependent metrics (Precision/Recall/F1/Accuracy) assume your default `threshold` (often 0.50).
- Champion selection and threshold policy are documented in `artifacts/champion/`.
