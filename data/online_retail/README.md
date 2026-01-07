## Online Retail II Dataset

**Source:**  
https://www.kaggle.com/datasets/mashlyn/online-retail-ii-uci

**Original Source:**  
UCI Machine Learning Repository

**Description:**  
This dataset contains transactional data from an online retail store, including invoice-level purchase records, product details, quantities, prices, and timestamps. It represents customer purchasing behavior over time.

**Size:**  
~1 million transaction records  
Time span: multiple years

**Key Features:**
- Invoice number
- Product description
- Quantity and unit price
- Invoice date (timestamp)
- Customer ID
- Country

**Why this dataset was chosen:**  
This dataset provides **event-based, temporal behavioral data**, which is highly relevant for understanding:
- customer engagement patterns
- frequency and recency signals
- session aggregation techniques

These concepts directly inform feature engineering strategies for churn prediction in subscription-based systems.

**Data Collection Method:**  
Downloaded from Kaggle as a CSV file.  
The dataset is small enough to be included directly in the repository.

**Usage in this project:**  
- Practice temporal aggregation and feature engineering
- Explore behavioral patterns that precede disengagement
- Supplementary analysis alongside the primary churn dataset

**Notes:**  
This dataset is used for exploratory and feature engineering insights only. The final churn model is trained on the KKBox dataset, which includes richer user activity logs and subscription lifecycle data.
