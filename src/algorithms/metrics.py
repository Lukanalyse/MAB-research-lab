from __future__ import annotations

import pandas as pd


def summary_table(df: pd.DataFrame) -> pd.DataFrame:
    """Create a compact summary table for one or several policy simulations."""
    required_columns = {"policy", "reward", "cumulative_reward", "cumulative_regret"}
    missing_columns = required_columns.difference(df.columns)
    if missing_columns:
        raise ValueError(f"Missing required columns: {sorted(missing_columns)}")

    rows = []

    for policy, group in df.groupby("policy", sort=False):
        last = group.iloc[-1]
        count_cols = [col for col in group.columns if col.startswith("count_arm_")]

        if count_cols:
            final_counts = last[count_cols].astype(float)
            most_pulled_arm = int(final_counts.idxmax().replace("count_arm_", ""))
            total_pulls = int(final_counts.sum())
        else:
            most_pulled_arm = -1
            total_pulls = len(group)

        rows.append(
            {
                "policy": policy,
                "final_cumulative_reward": float(last["cumulative_reward"]),
                "final_cumulative_regret": float(last["cumulative_regret"]),
                "most_pulled_arm": most_pulled_arm,
                "total_pulls": total_pulls,
                "mean_reward": float(group["reward"].mean()),
            }
        )

    return pd.DataFrame(rows)
