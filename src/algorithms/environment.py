from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class BanditArmInfo:
    """Description of one arm in a Gaussian multi-armed bandit."""

    arm: int
    mean: float
    std: float


class GaussianBandit:
    """Gaussian K-armed bandit environment.

    The environment contains K arms. Each arm a generates rewards according to
    a Gaussian distribution with true mean mu_a and standard deviation sigma_a:

        X_{a,t} ~ Normal(mu_a, sigma_a^2)

    The true means are used only by the simulator to compute regret. The policy
    does not observe them directly.
    """

    def __init__(self, means: Sequence[float], stds: Sequence[float], seed: int | None = None) -> None:
        self.means = np.asarray(means, dtype=float)
        self.stds = np.asarray(stds, dtype=float)

        if self.means.ndim != 1 or self.stds.ndim != 1:
            raise ValueError("means and stds must be one-dimensional sequences.")

        if len(self.means) != len(self.stds):
            raise ValueError("means and stds must have the same length.")

        if len(self.means) < 2:
            raise ValueError("A bandit environment must contain at least two arms.")

        if np.any(self.stds <= 0):
            raise ValueError("All standard deviations must be strictly positive.")

        self.rng = np.random.default_rng(seed)

    @property
    def n_arms(self) -> int:
        """Number of available arms."""
        return int(len(self.means))

    @property
    def optimal_arm(self) -> int:
        """Index of the arm with the largest true expected reward."""
        return int(np.argmax(self.means))

    @property
    def optimal_mean(self) -> float:
        """True mean reward of the optimal arm."""
        return float(np.max(self.means))

    def pull(self, arm: int) -> float:
        """Sample one reward from the selected arm."""
        if arm < 0 or arm >= self.n_arms:
            raise ValueError(f"Invalid arm index: {arm}. Expected 0 <= arm < {self.n_arms}.")

        return float(self.rng.normal(loc=self.means[arm], scale=self.stds[arm]))

    def arm_table(self) -> list[BanditArmInfo]:
        """Return a readable description of all arms."""
        return [
            BanditArmInfo(arm=i, mean=float(self.means[i]), std=float(self.stds[i]))
            for i in range(self.n_arms)
        ]
