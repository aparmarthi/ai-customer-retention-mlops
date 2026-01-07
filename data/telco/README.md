## Telco Customer Churn Dataset

**Source:**  
https://www.kaggle.com/datasets/blastchar/telco-customer-churn

**Description:**  
This dataset contains customer-level information from a telecommunications company, including demographics, account information, subscribed services, and churn labels. The objective is to predict whether a customer will churn (cancel service).

**Size:**  
~7,000 customer records  
~20 features

**Key Features:**
- Customer demographics (gender, senior citizen status)
- Account information (tenure, contract type, payment method)
- Service usage (internet service, phone service, add-ons)
- Target label: `Churn` (Yes / No)

**Why this dataset was chosen:**  
This dataset serves as a **baseline churn prediction reference** and helps validate feature engineering ideas commonly used in subscription and CRM-based churn modeling. While smaller than the primary dataset, it provides a clean, well-understood structure for early exploration and comparison.

**Data Collection Method:**  
Downloaded directly from Kaggle as a CSV file.  
Due to its small size, the full dataset is included in this repository.

**Usage in this project:**  
- Exploratory data analysis (EDA)
- Feature engineering inspiration
- Baseline churn modeling comparison

**Notes:**  
This dataset is not used as the final training dataset for the capstone model. The primary dataset for modeling is the KKBox Churn Prediction dataset, which better reflects real-world scale and complexity.
