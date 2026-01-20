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