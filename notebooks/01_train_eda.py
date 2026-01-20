import pandas as pd
import matplotlib.pyplot as plt
# Simple exploratory script for the KKBox training data
# - loads the CSV into a DataFrame
# - prints basic DataFrame info and the first rows
# - computes churn class counts and churn rate


# path to the raw training CSV (relative to project root)
train_path = "data/kkbox/raw/train.csv"
# load the training data into a DataFrame
train = pd.read_csv(train_path)
# display basic info about the DataFrame
train.info()
# display the first few rows of the DataFrame
train.head()

# compute churn class counts
churn_counts = train["is_churn"].value_counts()
# compute churn rate                            
churn_rate = churn_counts/len(train)
print("Churn Counts:\n", churn_counts)
print("Churn Rate:\n", churn_rate)

#is msno unique?
total_rows = len(train)
unique_users = train['msno'].nunique()

print(f'Total Rows: {total_rows}')
print(f'Unique Users: {unique_users}')

if total_rows == unique_users:
    print('msno is unique for each row.')   
else:
    print('msno is NOT unique for each row.')

#Any duplicate lables for the same user?

label_variation = (train.groupby('msno')['is_churn']
                   .nunique()
                   .reset_index(name='label_count')
                   )
conflicting_labels = label_variation[label_variation['label_count'] > 1]
print(f'Number of users with conflicting labels: {len(conflicting_labels)}')
conflicting_labels.head()

churn_rate_pct = churn_rate * 100
churn_rate_pct

churn_counts.plot(kind="bar")
plt.title("Churn Distribution")
plt.xlabel("is_churn")
plt.ylabel("Count")
plt.show()

#Dataset is imbalanced, requiring appropriate evaluation metrics (AUC, F1, PR-AUC).
train.columns
train.is_churn.describe()
train.is_churn.value_counts(normalize=True)

print("""
TRAIN DATASET – KEY FINDINGS
1. Churn rate is imbalanced, with fewer churned users than retained users.
2. Each msno appears once, indicating a single label per user.
3. No conflicting churn labels were found.
4. Churn labels are provided without explicit temporal boundaries, implying an externally defined churn window.
5. Class imbalance suggests using metrics beyond accuracy (ROC-AUC, F1, PR-AUC).
""")