from __future__ import annotations

import pandas as pd


def summary_table(df: pd.DataFrame) -> pd.DataFrame:
    """Create a compact summary table for one or several policy simulations."""
    required_columns = {
        "policy",
        "selected_arm",
        "reward",
        "cumulative_reward",
        "cumulative_regret",
        "optimal_arm",
    }
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

        optimal_arm = int(last["optimal_arm"])
        final_selected_arm = int(last["selected_arm"])
        share_optimal_arm = float((group["selected_arm"] == optimal_arm).mean())
        final_cumulative_reward = float(last["cumulative_reward"])
        final_cumulative_regret = float(last["cumulative_regret"])

        rows.append(
            {
                "policy": policy,
                "final_cumulative_reward": final_cumulative_reward,
                "final_cumulative_regret": final_cumulative_regret,
                "most_pulled_arm": most_pulled_arm,
                "total_pulls": total_pulls,
                "mean_reward": float(group["reward"].mean()),
                "optimal_arm": optimal_arm,
                "final_selected_arm": final_selected_arm,
                "share_optimal_arm": share_optimal_arm,
                "is_most_pulled_arm_optimal": most_pulled_arm == optimal_arm,
                "final_average_reward": final_cumulative_reward / total_pulls,
                "final_average_regret": final_cumulative_regret / total_pulls,
            }
        )

    return pd.DataFrame(rows)
