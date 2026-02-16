Capstone Step 5: Data Wrangling & Exploratory Data Analysis Summary

Project: Early Churn Prediction for Subscription-Based Digital Services (KKBox)

1. Overview

The goal of this step was to clean, wrangle, and explore the KKBox churn dataset to prepare a model-ready, leakage-safe feature table suitable for time-aware machine learning.

Given the large scale (millions of rows across multiple relational tables), emphasis was placed on:

Understanding dataset structure and quality

Identifying churn-relevant signals

Designing aggregation strategies rather than row-level modeling

Preventing data leakage through temporal awareness

The primary datasets explored include:

train.csv (churn labels)

members_v3.csv (user metadata)

transactions.csv (subscription behavior)

user_logs.csv (engagement activity)

2. Data Cleaning & Wrangling Steps
2.1 Training Labels (train.csv)

Key checks performed:

Verified class balance and churn rate

Confirmed label uniqueness per user

Checked for conflicting labels

Findings:

Dataset is highly imbalanced, with churners representing ~6% of users

Each msno appears exactly once, confirming one label per user

No conflicting churn labels were found

Implications:

Accuracy is not an appropriate evaluation metric

Downstream modeling will emphasize ROC-AUC, PR-AUC, and top-K targeting metrics

📄 Source: 01_train_eda.py 

01_train_eda

2.2 Member Metadata (members_v3.csv)

Cleaning considerations:

Missing and noisy demographic fields

Outlier detection for age

Categorical skew

Key issues identified:

bd (age) contains unrealistic values (<10 or >100)

gender contains a high proportion of missing values

City and registration channel distributions are heavily skewed

Wrangling decisions:

Invalid ages flagged and excluded or capped

Gender treated as optional categorical feature (or encoded with “unknown”)

Demographic features treated cautiously to avoid proxy bias

Rationale:
Demographic features can be predictive but may introduce fairness and ethical concerns. These signals are retained but carefully evaluated downstream.

📄 Source: 02_members3_eda.py 

02_members3_eda

2.3 Transactions (transactions.csv)

Scale & structure:

Tens of millions of rows

Multiple transactions per user

Strong temporal component

Wrangling approach:

Chunk-based processing to avoid memory overload

No row-level modeling

Aggregation to per-user features only

Key churn-relevant signals identified:

Auto-renew status

Cancellation flags

Plan price changes

Transaction frequency and tenure

Critical leakage control:

All transactional features will be computed strictly prior to the churn label window

Raw transactions are never joined directly to the label table

📄 Source: 03_transactions_eda.py 

03_transactions_eda

2.4 User Engagement Logs (user_logs.csv)

Scale & challenges:

Extremely large daily-level activity dataset

Not feasible for full in-memory processing

Wrangling strategy:

Header inspection and chunked scanning only

Confirmed availability of engagement signals such as:

Total listening time

Unique content consumption

Playback completion buckets

Design decision:

Raw logs are not suitable for direct modeling

Engagement data will be aggregated into rolling, time-bounded features:

Recency

Frequency

Activity trends

Engagement decay

Leakage prevention:

Aggregations aligned to fixed cutoffs prior to churn labels

📄 Source: 04_user_logs_eda.py 

04_user_logs_eda

3. Handling Missing Values
Dataset	Field	Strategy
members_v3	bd (age)	Remove or cap unrealistic values
members_v3	gender	Encode as categorical with missing/unknown
transactions	N/A	Aggregated; missing handled via aggregation defaults
user_logs	N/A	Aggregated; missing days imply inactivity

Missing values were handled contextually, not globally, based on domain relevance and downstream modeling needs.

4. Outliers

Age outliers identified and excluded/corrected

Engagement outliers retained but handled via aggregation (mean, max, recency)

Transaction count outliers preserved as potential churn signals

Outliers were not blindly removed; instead, their business meaning was evaluated before deciding treatment.

5. Dataset Size & Sampling Strategy

Given the size of the KKBox dataset:

Full datasets are processed via chunking and aggregation

Modeling will occur on a per-user feature table

Small sampled extracts are retained for rapid experimentation and reproducibility

This balances real-world scale realism with practical development constraints.

6. Optional Exploratory Data Analysis (EDA)
Key insights:

Churn is rare but systematic, reinforcing the need for ranking-based evaluation

Transactional behaviors (auto-renew, cancellations) are strong churn indicators

Engagement decay patterns are likely more informative than raw volume

Demographic features show skew and potential bias risk

These insights directly inform:

Feature engineering choices

Evaluation metric selection

Ethical considerations in model usage

7. Outcome of Step 5

At the end of this step:

The dataset is cleaned, understood, and structured

Clear aggregation and leakage-safe feature strategies are defined

A model-ready per-user table can now be constructed confidently

The groundwork is laid for time-aware churn modeling and MLOps-style pipelines

This completes the Data Wrangling and EDA phase and prepares the project for Step 6: Feature Engineering and Model Prototyping.

What to submit (checklist)