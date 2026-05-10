from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

from src.algorithms.environment import GaussianBandit
from src.algorithms.policies import make_policy


def default_gaussian_bandit_parameters(n_arms: int) -> tuple[np.ndarray, np.ndarray]:
    """Return simple default means and standard deviations for a Gaussian bandit.

    The means are ordered increasingly so the last arm is optimal. This makes the
    behavior of random allocation and UCB easy to interpret in the Streamlit app.
    """
    if n_arms < 2:
        raise ValueError("n_arms must be at least 2.")

    means = np.linspace(0.40, 0.70, n_arms)
    stds = np.linspace(0.10, 0.25, n_arms)
    return means, stds


def run_simulation(
    policy_name: str,
    means: Sequence[float],
    stds: Sequence[float],
    horizon: int,
    seed: int = 42,
) -> pd.DataFrame:
    """Run one stochastic Gaussian bandit simulation.

    Parameters
    ----------
    policy_name:
        Name of the policy to run. Currently supported: "random" and "ucb".
    means:
        True mean reward of each arm.
    stds:
        Standard deviation of each arm.
    horizon:
        Number of sequential pulls.
    seed:
        Random seed used for reproducibility.

    Returns
    -------
    pandas.DataFrame
        One row per time step, with rewards, regret, empirical means, and pull
        counts. Regret is computed from the true means, which are known to the
        simulator but not to the policy.
    """
    if horizon < 1:
        raise ValueError("horizon must be at least 1.")

    env = GaussianBandit(means=means, stds=stds, seed=seed)
    policy = make_policy(policy_name=policy_name, n_arms=env.n_arms, seed=seed + 1)

    cumulative_reward = 0.0
    cumulative_regret = 0.0
    rows: list[dict[str, float | int | str]] = []

    for t in range(1, horizon + 1):
        selected_arm = policy.select_arm(t)
        reward = env.pull(selected_arm)
        policy.update(selected_arm, reward)

        instantaneous_regret = env.optimal_mean - float(env.means[selected_arm])
        cumulative_reward += reward
        cumulative_regret += instantaneous_regret

        row: dict[str, float | int | str] = {
            "t": t,
            "policy": policy.name,
            "selected_arm": selected_arm,
            "reward": reward,
            "cumulative_reward": cumulative_reward,
            "instantaneous_regret": instantaneous_regret,
            "cumulative_regret": cumulative_regret,
            "optimal_arm": env.optimal_arm,
            "optimal_mean": env.optimal_mean,
        }

        for arm in range(env.n_arms):
            row[f"empirical_mean_arm_{arm}"] = float(policy.values[arm])
            row[f"count_arm_{arm}"] = int(policy.counts[arm])
            row[f"true_mean_arm_{arm}"] = float(env.means[arm])

        rows.append(row)

    return pd.DataFrame(rows)


def run_comparison(
    policy_names: Sequence[str],
    means: Sequence[float],
    stds: Sequence[float],
    horizon: int,
    seed: int = 42,
) -> pd.DataFrame:
    """Run several policies on the same bandit parameters."""
    frames = []

    for index, policy_name in enumerate(policy_names):
        frame = run_simulation(
            policy_name=policy_name,
            means=means,
            stds=stds,
            horizon=horizon,
            seed=seed + 10_000 * index,
        )
        frames.append(frame)

    return pd.concat(frames, ignore_index=True)
