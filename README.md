🚀 AI Customer Retention & Decision Intelligence Platform
End-to-End ML System for Large-Scale Subscription Churn Prediction

Production-oriented Machine Learning system built on ~28GB of real-world subscription data (KKBox) to predict churn, optimize intervention strategy, and simulate measurable business ROI.

This project goes beyond model accuracy and demonstrates full-stack ML system design, governance, deployment, and business alignment.

🧭 Executive Overview

This platform answers three critical questions for subscription businesses:

Who is likely to churn?

Which customers should we target given budget constraints?

What is the projected ROI of intervention?

It integrates:

Scalable feature engineering

Imbalanced classification at million-row scale

MLflow-based experiment governance

Leaderboard-driven model selection

Threshold + Top-K decision policy engine

ROI simulation framework

FastAPI production inference layer

Streamlit executive analytics dashboard

This mirrors real-world MLE + MLOps + AI Product workflows.

🧠 Business Context

Subscription businesses operate under:

Low churn base rate (~6%)

Limited retention budgets

High intervention costs

Need for precision targeting

Traditional ML projects optimize ROC-AUC.
This system optimizes business ROI.

We explicitly connect:

Model Probability → Decision Policy → Financial Impact
📦 Dataset
Primary Dataset
KKBox Churn Prediction Dataset

~28GB subscription & transaction logs

~1M+ rows in processed model table

Highly imbalanced (~6% churn)

Time-dependent behavioral signals

Data Engineering Challenges Solved

Large-scale aggregation

Time-aware feature generation

Prevention of data leakage

Memory-aware processing

Reproducible data pipeline scripts

Sample dataset generation for lightweight experimentation

Large raw files excluded from Git.
Model-ready samples included for reproducibility.

🏗 End-to-End System Architecture
Raw Logs
   ↓
ETL & Aggregation
   ↓
Feature Engineering
   ↓
Time-aware Train/Validation Split
   ↓
Model Training (LR / RF / XGB / LGBM)
   ↓
MLflow Experiment Tracking
   ↓
Leaderboard Generation
   ↓
Decision Policy Layer (Threshold / Top-K)
   ↓
ROI Simulation Engine
   ↓
FastAPI Inference Service
   ↓
Streamlit Executive Dashboard

Designed to mirror production ML architecture.

🔄 Data Pipeline (src/data/)

Key capabilities:

Raw ingestion & cleaning

User-level aggregation

Behavioral feature engineering

Recency / tenure / frequency metrics

Target generation

Model table creation

Time-based splitting

Sample extraction for testing

Leakage prevention via time-aware split.

All transformations are scripted — no manual notebook dependency.

🤖 Modeling
Implemented Models

Majority Class Baseline

Logistic Regression

Random Forest

XGBoost

LightGBM (Champion candidate)

Observed Results (Best Model Example – XGBoost)

ROC-AUC: 0.9875

PR-AUC: 0.8771

Recall: 0.94

F1: 0.71

Strong performance on imbalanced data

Imbalance Strategy

scale_pos_weight

PR-AUC driven evaluation

Threshold optimization

Business-aware targeting

🏆 Model Governance & Leaderboard

Automated leaderboard generation via:

scripts/generate_leaderboard.py

Features:

Aggregates all run logs

Ranks models by selected metric

Exports leaderboard.md

Enables transparent model comparison

Documents performance trade-offs

Simulates internal ML review workflow.

🎯 Decision Policy Engine

Located in src/serving/policy.py

Supports:

Threshold-based targeting

Top-K targeting

Rank-aware decisioning

ROI-aligned probability cutoffs

Fallback logic for single prediction cases

Outputs structured PolicyDecision objects including:

Churn probability

Action (target / no_target)

Policy used

Threshold

Rank (if applicable)

Decision metadata

This reflects production decision-layer design beyond raw probabilities.

💰 ROI Simulation Engine

Bridges ML predictions to financial outcomes.

Inputs:

Predicted probabilities

Cost per intervention

Revenue per retained user

Retention uplift rate

Targeting strategy

Outputs:

Revenue saved

Intervention cost

Net ROI

Optimal threshold

Sensitivity analysis

This is the AI Product layer of the system.

📊 Experiment Tracking (MLflow)

Tracks:

Hyperparameters

Evaluation metrics

Artifacts

Confusion matrices

Feature importances

Champion model bundle

Champion artifacts saved under:

artifacts/champion/
    threshold.json
    metrics.json
    feature_list.json
    flaml_best_params.json
    notes.md

Simulates production model registry workflow.

🌐 Deployment Layer
FastAPI Inference Service

REST endpoint

JSON in / JSON out

Batch and single prediction support

Policy decision integration

Production-style separation of concerns

Run via:

uvicorn app:app --reload
Streamlit Executive Dashboard

Provides:

Model performance visualization

Precision-Recall trade-off explorer

Threshold tuning interface

ROI scenario simulator

Business-facing analytics

Bridges technical and executive stakeholders.

📁 Repository Structure
ai-customer-retention-mlops/
│
├── data/
├── notebooks/
├── reports/
├── scripts/
├── artifacts/
│   └── champion/
├── src/
│   ├── data/
│   ├── models/
│   ├── serving/
│   ├── utils/
│
├── leaderboard.md
├── requirements.txt
└── README.md

Clean separation of:

Data

Code

Artifacts

Governance outputs

⚙️ How to Run

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
🧩 Production-Readiness Considerations

Time-aware splits

Leakage prevention

Artifact versioning

Model governance

ROI-based threshold selection

Modular architecture

Scalable design

Ready for Dockerization

Designed for CI/CD extension

🎯 What This Demonstrates
For Machine Learning Engineer Roles

Large-scale tabular modeling

Imbalanced classification

Feature engineering at scale

Model comparison workflows

Decision-layer modeling

For MLOps Roles

MLflow experiment tracking

Model artifact management

Champion model bundle

API deployment

Reproducibility

Structured project architecture

For AI Product Roles

Connecting model output to business value

ROI-based decision optimization

Threshold trade-off modeling

System-level thinking

End-to-end ML product lifecycle

🔮 Planned Enhancements

Docker containerization

CI/CD pipelines

Feature store abstraction

Drift detection

Monitoring dashboards

Scheduled retraining workflows

Cloud-native deployment (AWS/GCP)

🏁 Summary

This is not a notebook-based ML project.

It is a production-oriented churn intelligence platform that integrates:

Data engineering

Applied ML

Governance

Deployment

Financial modeling

Product decision logic

Built to reflect real-world ML system ownership from data ingestion to business impact.