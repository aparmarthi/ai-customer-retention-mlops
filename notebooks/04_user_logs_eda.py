"""
KKBox Churn Capstone – High-Level User Logs EDA

Purpose:
- Understand scale and structure of user_logs.csv
- Identify available user behaviors
- Confirm time coverage
- Explicitly define aggregation strategy (no row-level modeling)

IMPORTANT:
- This script intentionally avoids loading the full dataset into memory.
"""

import sys
import pandas as pd


# Interpreter sanity check
print("Python executable:", sys.executable)
print("Python version:", sys.version)
print("-" * 80)

# File path + chunk config
USER_LOGS_PATH = "data/kkbox/raw/user_logs.csv"
CHUNK_SIZE = 1_000_000   # safe chunk size

# Read header only (columns)
print("Inspecting user_logs.csv columns (header only)...")
columns = pd.read_csv(USER_LOGS_PATH, nrows=0).columns
print("Columns:")
for col in columns:
    print(f" - {col}")
print("-" * 80)

# Read first chunk for value intuition
print("Reading first chunk for sample values...")
chunk_iter = pd.read_csv(USER_LOGS_PATH, chunksize=CHUNK_SIZE)
first_chunk = next(chunk_iter)

print("Sample rows:")
print(first_chunk.head())
print("-" * 80)


#Dataset scale and date range (chunked)
print("Scanning user_logs.csv for scale and date range...")

total_rows = 0
unique_users = set()
min_date = None
max_date = None

for i, chunk in enumerate(pd.read_csv(USER_LOGS_PATH, chunksize=CHUNK_SIZE)):
    total_rows += len(chunk)

    # Track unique users (approximate)
    unique_users.update(chunk["msno"].unique())

    # Track date range
    if "date" in chunk.columns:
        chunk_min = chunk["date"].min()
        chunk_max = chunk["date"].max()
        min_date = chunk_min if min_date is None else min(min_date, chunk_min)
        max_date = chunk_max if max_date is None else max(max_date, chunk_max)

    # Stop early – high-level EDA only
    if i == 4:
        break

print(f"Rows scanned (approx): {total_rows:,}")
print(f"Unique users seen (approx): {len(unique_users):,}")

if min_date and max_date:
    print(f"Date range (approx): {min_date} → {max_date}")

print("-" * 80)

# Behaviors overview
print("Behaviors overview:")

behavior_columns = [
    "num_25",
    "num_50",
    "num_75",
    "num_985",
    "num_100",
    "num_unq",
    "total_secs"
]

for col in behavior_columns:
    if col in first_chunk.columns:
        print(f" - {col}: present")
    else:
        print(f" - {col}: NOT present")

print("-" * 80)


# Sanity checks on engagement metrics
print("Sanity check on engagement metrics (first chunk only):")

if "total_secs" in first_chunk.columns:
    print("total_secs summary:")
    print(first_chunk["total_secs"].describe())

if "num_unq" in first_chunk.columns:
    print("\nnum_unq summary:")
    print(first_chunk["num_unq"].describe())

print("-" * 80)


# Explicit aggregation plan (EDA conclusion)
# ------------------------------------------
print(
    """
========================
USER_LOGS – EDA SUMMARY
========================

1. user_logs.csv is a very large, time-series dataset with multiple rows per user.
2. The dataset contains daily-level engagement signals such as:
   - total listening time (total_secs)
   - unique songs (num_unq)
   - play count buckets (num_25, num_50, num_75, num_985, num_100)
3. Raw log rows are NOT suitable for direct modeling due to size and temporal granularity.
4. Engagement data must be aggregated to per-user features over fixed windows.

PLANNED AGGREGATIONS (examples):
- Total listening time (sum, mean)
- Active days count
- Recency of last activity
- Engagement trends (recent vs historical)
- Unique content consumption

LEAKAGE WARNING:
- All aggregates must be computed strictly from dates PRIOR to the churn label window.

NEXT STEP:
- Build per-user aggregated features from user_logs aligned to the training cutoff.
"""
)


