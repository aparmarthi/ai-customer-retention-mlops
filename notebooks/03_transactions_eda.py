"""
Docstring for notebooks.04_transactions_eda
High level transactions EDA
Purpose:
- Understand scale and structure of transactions.csv
- Identify churn-relevant signals (renewals, cancellations, pricing)
- Avoid full in-memory loads and leakage
"""
import sys
import pandas as pd
from collections import Counter
from datetime import datetime

print("Python executable:", sys.executable)
print("Python version:", sys.version)
print("-" * 80)

TRANSACTIONS_PATH = "data/kkbox/raw/transactions.csv"
chunk_size = 1_000_000  # Adjust chunk size based on memory capacity
print(f"Loading transactions.csv in chunks of {chunk_size} rows...")

# Read a small sample to inspect structure
df_sample = pd.read_csv("data/kkbox/raw/transactions.csv", nrows=5)

print(df_sample.head())
print(df_sample.columns)

#Initialize counters and accumulators
total_rows = 0
unique_users = set()

#Track min and max dates
min_date = None
max_date = None

#Track Distributions
payment_methods = Counter()
plan_prices = Counter()
auto_renewals = Counter()
cancel_counts = Counter()

#Track transactions per user
transactions_count_per_user = Counter()

#Process file in chunks
print("Starting chunked scan of transactions.csv ...")
for i, chunk in enumerate(pd.read_csv(TRANSACTIONS_PATH, chunksize=chunk_size)):
    total_rows += len(chunk)
    
    # Unique users
    users = chunk["msno"].unique()
    unique_users.update(users)

    # Transactions per user
    transactions_count_per_user.update(chunk["msno"].value_counts().to_dict())
    
    # Payment method & pricing signals
    payment_methods.update(chunk["payment_method_id"].value_counts().to_dict())
    plan_prices.update(chunk["plan_list_price"].value_counts().to_dict())

    # Auto-renew & cancellation signals
    auto_renewals.update(chunk["is_auto_renew"].value_counts().to_dict())
    cancel_counts.update(chunk["is_cancel"].value_counts().to_dict())

     # Date range (transaction_date usually YYYYMMDD int)
    if "transaction_date" in chunk.columns:
        chunk_min = chunk["transaction_date"].min()
        chunk_max = chunk["transaction_date"].max()

        min_date = chunk_min if min_date is None else min(min_date, chunk_min)
        max_date = chunk_max if max_date is None else max(max_date, chunk_max)
    if (i + 1) % 5 == 0:
        print(f"Processed {(i + 1) * chunk_size:,} rows...")

print("-" * 80)

#High Level dataset
print("TRANSACTIONS DATASET OVERVIEW")
print(f"Total transaction rows: {total_rows:,}")
print(f"Unique users (msno): {len(unique_users):,}")
print(f"Avg transactions per user: {total_rows / len(unique_users):.2f}")
print("-" * 80)

# Date range coverage
if min_date and max_date:
    print("Transaction date range:")
    print(f"Min date: {min_date}")
    print(f"Max date: {max_date}")
print("-" * 80)

# Payment method distribution (top 10)
print("Top 10 payment methods:")
for pm, cnt in payment_methods.most_common(10):
    print(f"Payment method {pm}: {cnt:,}")

print("-" * 80)

# Plan price distribution (top 10)
print("Top 10 plan prices:")
for price, cnt in plan_prices.most_common(10):
    print(f"Plan price {price}: {cnt:,}")

print("-" * 80)

# Auto-renew and cancellation signals
print("Auto-renew distribution:")
for k, v in auto_renewals.items():
    print(f"is_auto_renew={k}: {v:,}")

print("\nCancellation distribution:")
for k, v in cancel_counts.items():
    print(f"is_cancel={k}: {v:,}")

print("-" * 80)

# Transactions per user (distribution insight)
txn_counts_series = pd.Series(transactions_count_per_user.values())

print("Transactions per user – summary:")
print(txn_counts_series.describe())

print("-" * 80)

# Step 11) EDA summary (mentor-facing)
# ------------------------------------------
print(
    """
========================
TRANSACTIONS - EDA SUMMARY
========================

1. transactions.csv contains multiple rows per user, requiring aggregation before modeling.
2. Average transactions per user indicates varying subscription activity and tenure.
3. The dataset spans a broad time range, requiring careful alignment with churn label windows.
4. Payment method and plan price distributions are highly skewed.
5. Auto-renew and cancellation flags provide strong churn-related signals.
6. Direct use of raw transaction rows risks label leakage; features must be computed only from pre-label periods.

NEXT STEP:
- Aggregate transactions to per-user features (recency, frequency, tenure, renewal behavior).
- Then perform a light structural overview of user_logs.csv.
"""
)