
# AI Customer Retention & MLOps Platform

End-to-end Machine Learning system designed to predict subscription churn at scale and simulate business ROI impact.

Built on ~28GB of real-world subscription data, this project demonstrates scalable data engineering, applied machine learning, experiment tracking, deployment, and business-aligned decision modeling.

---

## 🚀 Executive Summary

This project goes beyond model accuracy. It demonstrates:

- Large-scale ETL & feature engineering
- Imbalanced classification at production scale
- Model governance via MLflow
- Leaderboard-based model comparison
- API-based inference (FastAPI)
- Business impact modeling (ROI simulator)
- Deployment-ready architecture
- Clean separation of code, data, and artifacts

Designed to reflect real-world MLE, MLOps, and AI Product workflows.

---

## 🧠 Business Problem

Subscription businesses lose significant revenue due to customer churn.

This system answers:

- Which customers are most likely to churn?
- How confident is each prediction?
- What threshold maximizes business ROI?
- What is the projected revenue impact of intervention?

The project integrates predictive modeling with decision optimization.

---

## 📦 Datasets

### Primary Dataset
- **KKBox Churn Prediction Dataset**
- ~28GB subscription and user behavior logs
- Millions of records
- Heavy feature engineering
- Imbalanced target (~6% churn)

### Secondary Datasets (Exploratory)
- Telco Customer Churn Dataset
- Online Retail II Dataset

Large datasets are excluded from Git. Sample subsets are included for reproducibility.

---

## 🏗 System Architecture

Raw Data → ETL & Aggregation → Feature Engineering → Model Training → MLflow Tracking → Leaderboard → ROI Simulation → API Service → Interactive Dashboard

This mirrors real production ML pipelines.

---

## 🔄 Data Pipeline

Located in `src/data/`

Pipeline stages:
- Raw ingestion
- Log aggregation
- Feature engineering (recency, tenure, frequency, behavioral metrics)
- Derived table creation
- Model-ready dataset generation
- Sample data generation for lightweight experimentation

Key considerations:
- Time-aware splitting to prevent leakage
- Reproducibility through scripted pipelines
- Scalability for GB-scale processing

---

## 🤖 Modeling Approach

Models implemented:

- Logistic Regression (baseline)
- Random Forest
- XGBoost
- LightGBM (primary production candidate)

Evaluation metrics:

- ROC-AUC
- PR-AUC (critical for imbalanced data)
- Precision / Recall
- F1 Score
- Threshold-based confusion analysis

Imbalance handled via:
- Scale_pos_weight
- PR-driven threshold selection
- Business-aligned decision thresholds

---

## 📊 Model Governance & Leaderboard

`scripts/generate_leaderboard.py`

Features:
- Aggregates experiment results
- Ranks models by selected metric
- Outputs `leaderboard.md`
- Documents performance trade-offs

Simulates internal ML governance workflows.

---

## 💰 ROI Simulation Engine

Bridges ML predictions to financial outcomes.

Inputs:
- Predicted churn probabilities
- Intervention cost assumptions
- Retention uplift rate

Outputs:
- Estimated revenue saved
- Net ROI
- Sensitivity analysis

Demonstrates alignment between ML systems and business impact.

---

## 📈 Experiment Tracking (MLflow)

- Hyperparameter tracking
- Metric logging
- Artifact storage
- Champion model selection
- Reproducible runs

Reflects real-world MLOps practices.

---

## 🌐 Deployment Layer

### FastAPI Inference Service
- REST endpoint for predictions
- JSON request/response
- Production-style structure

### Streamlit Dashboard
- Model performance visualization
- Threshold tuning interface
- ROI scenario exploration
- Executive-friendly analytics layer

---

## 📁 Repository Structure

ai-customer-retention-mlops/
│
├── data/ (excluded large datasets)
├── notebooks/
├── reports/
├── scripts/
├── src/
│   ├── data/
│   ├── models/
│   ├── utils/
├── leaderboard.md
└── README.md

Designed for clarity, modularity, and cloud portability.

---

## ⚙️ How to Run

Install dependencies:

pip install -r requirements.txt

Run data pipeline:

python src/data/04_aggregate_user_logs.py

Train model:

python src/models/04_xgboost.py

Generate leaderboard:

python scripts/generate_leaderboard.py

Start API:

uvicorn app:app --reload

Launch dashboard:

streamlit run dashboard.py

---

## 🧩 Production Considerations

- Separation of data vs code
- Feature leakage prevention
- Threshold optimization by ROI
- Reproducible pipelines
- Scalable architecture design
- Clean artifact management
- Designed for containerization and cloud deployment

---

## 🎯 What This Project Demonstrates

For MLE:
- Large-scale tabular ML
- Imbalanced modeling
- End-to-end system design

For MLOps:
- Experiment tracking
- Model comparison workflows
- Deployment-ready APIs
- Reproducibility & artifact management

For AI PM:
- Business metric alignment
- ROI modeling
- Threshold decision trade-offs
- System-level thinking beyond model accuracy

---

## 🔮 Future Enhancements

- Dockerization
- CI/CD integration
- Feature store abstraction
- Drift detection & monitoring
- Automated retraining workflows
- Cloud-native deployment

---

Built to demonstrate production-grade ML system design and measurable business impact.