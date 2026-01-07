"""
Data collection and preparation script.

Primary dataset:
- KKBox Churn Prediction Dataset (Kaggle)
Secondary datasets:
- Telco Customer Churn (Kaggle)
- Online Retail II (UCI / Kaggle)

Notes:
- Full dataset is downloaded manually from Kaggle due to licensing.
- Raw data is kept locally and not committed to GitHub.
- This script creates small, reproducible samples for exploration
  and version control.
- Secondary datasets are not used in this project but are kept for reference.

print("See dataset READMEs for download links and instructions.")
"""

from pathlib import Path

import pandas as pd

RAW_PATH = Path("data/kkbox/raw")
SAMPLE_PATH = Path("data/kkbox/sample")
SAMPLE_PATH.mkdir(parents=True, exist_ok=True)


def create_train_sample(
    filename: str,
    sample_size: int = 5000,
    random_state: int = 42,
):
    """Create a reproducible sample CSV from the raw KKBox dataset."""
    input_file = RAW_PATH / filename
    output_file = SAMPLE_PATH / f"{filename.replace('.csv', '_sample.csv')}"

    df = pd.read_csv(input_file)
    df.sample(sample_size, random_state=random_state).to_csv(
        output_file,
        index=False,
    )

    print(f"Sample created: {output_file} ({sample_size} rows)")


if __name__ == "__main__":
    create_train_sample("train.csv")
