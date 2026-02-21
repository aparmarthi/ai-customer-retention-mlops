from pathlib import Path
import pandas as pd
from src.serving.policy import load_policy, decide_batch, decide_single_with_fallback

PROJECT_ROOT = Path(__file__).resolve().parents[2]
policy = load_policy(PROJECT_ROOT / "artifacts" / "champion" / "deployment_policy.json")

# Test single
print(decide_single_with_fallback(prob=0.72, policy=policy))
print(decide_single_with_fallback(prob=0.40, policy=policy))

# Test batch
df = pd.read_parquet(PROJECT_ROOT / "artifacts" / "champion" / "valid_scored.parquet").head(20000)
df2 = decide_batch(df, policy)
print(df2[["y_proba", "rank", "action", "action_threshold"]].head(10))
print("Targets via top-k:", (df2["action"] == "target").sum())
