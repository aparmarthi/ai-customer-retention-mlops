import pandas as pd
import matplotlib.pyplot as plt
import sys

print("Python executable:", sys.executable)
print("Python version:", sys.version)
print("-"*80)

MEMBERS_PATH = "data/kkbox/raw/members_v3.csv"
print("Loading members_v3.csv ...")
members = pd.read_csv(MEMBERS_PATH)

print("Dataset shape (rows, columns):", members.shape)
print("\n First 5 rows")
print(members.head())
print("-"*80)

# Review column names, data types, and non-null counts
print("Dataset info:")
members.info()
print("-" * 80)

#Row count & uniqueness of msno
total_rows = len(members)
unique_users = members["msno"].nunique()

print(f"Total rows: {total_rows}")
print(f"Unique msno: {unique_users}")

if total_rows == unique_users:
    print("✅ msno is unique (1 row per user)")
else:
    print("⚠️ Duplicate users detected")

#Missing values analysis
missing_pct = (
    members.isnull()
    .mean()
    .sort_values(ascending=False) * 100
)

print(missing_pct)

missing_summary = missing_pct.reset_index()
missing_summary.columns = ["column", "missing_percent"]
print(missing_summary)

#Age (bd) distribution & outliers
print("\n Age Distribution: \n", members["bd"].describe())
#check unrealistic ages
invalid_age = members[(members["bd"] < 10) | (members["bd"] > 100)]
print(f"\n Users with unrealistic ages: {len(invalid_age)}")

#Gender distribution
print("\n Gender Value Counts:" ,members["gender"].value_counts(dropna=False))
print("\n Gender % distribution", members["gender"].value_counts(normalize=True, dropna=False) * 100)

#City distribution (top cities only)
print("City Distribution: \n", members["city"].value_counts().head(10))
print("\n City Distribution Normalized:", members["city"].value_counts(normalize=True).head(10) * 100)

#Registration Channel (registered_via)
print("\n Registered via:", members["registered_via"].value_counts().sort_index())

#Registration date range
members["registration_init_time"] = pd.to_datetime(members["registration_init_time"], format="%Y%m%d")
print("\n Registration date range:", members["registration_init_time"].min(), "to", members["registration_init_time"].max())  

print("""
MEMBERS_V3 – KEY EDA FINDINGS

1. Dataset contains one row per user with unique msno.
2. Several demographic fields (gender, age) contain missing or unrealistic values.
3. Age (bd) has significant outliers, indicating self-reported or noisy data.
4. Gender information is incomplete, which may impact fairness analysis.
5. City and registration channel show strong skew, suggesting regional and acquisition-pattern effects.
6. Demographic features may introduce bias and must be handled carefully in modeling.
""")