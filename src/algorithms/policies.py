from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class PolicyState:
    """Current state of a bandit policy."""

    counts: np.ndarray
    values: np.ndarray
    total_reward: float


class BasePolicy:
    """Base class for stochastic multi-armed bandit policies.

    Every policy tracks how often each arm has been pulled and maintains an
    online empirical mean for each arm. The update formula is

        mean <- mean + (reward - mean) / n

    where n is the updated number of pulls for the selected arm.
    """

    name = "base"

    def __init__(self, n_arms: int, seed: int | None = None) -> None:
        if n_arms < 2:
            raise ValueError("A policy requires at least two arms.")

        self.n_arms = n_arms
        self.rng = np.random.default_rng(seed)
        self.counts = np.zeros(n_arms, dtype=int)
        self.values = np.zeros(n_arms, dtype=float)
        self.total_reward = 0.0

    @property
    def state(self) -> PolicyState:
        """Return a copy of the current policy state."""
        return PolicyState(
            counts=self.counts.copy(),
            values=self.values.copy(),
            total_reward=float(self.total_reward),
        )

    def select_arm(self, t: int) -> int:
        """Select one arm at time t."""
        raise NotImplementedError

    def update(self, arm: int, reward: float) -> None:
        """Update counts, empirical means, and cumulative reward."""
        if arm < 0 or arm >= self.n_arms:
            raise ValueError(f"Invalid arm index: {arm}.")

        self.counts[arm] += 1
        n = self.counts[arm]
        self.values[arm] += (reward - self.values[arm]) / n
        self.total_reward += reward


class RandomPolicy(BasePolicy):
    """Uniform random allocation baseline.

    At every time step, the selected arm is drawn uniformly from {0, ..., K-1}.
    This policy is not adaptive and does not use observed rewards to guide future
    choices. It is useful as a neutral benchmark.
    """

    name = "random"

    def select_arm(self, t: int) -> int:
        del t
        return int(self.rng.integers(0, self.n_arms))


class UCBPolicy(BasePolicy):
    """Upper Confidence Bound policy using the classical UCB1 score.

    The score of arm a at time t is

        UCB_a(t) = empirical_mean_a + sqrt(2 log(t) / N_a(t))

    Each arm is pulled once at the beginning to avoid division by zero. The rule
    implements optimism under uncertainty: rarely sampled arms receive a larger
    exploration bonus.
    """

    name = "ucb"

    def select_arm(self, t: int) -> int:
        for arm in range(self.n_arms):
            if self.counts[arm] == 0:
                return arm

        safe_t = max(t, 2)
        bonus = np.sqrt((2.0 * np.log(safe_t)) / self.counts)
        scores = self.values + bonus
        return int(np.argmax(scores))


def make_policy(policy_name: str, n_arms: int, seed: int | None = None) -> BasePolicy:
    """Factory function for supported bandit policies."""
    normalized_name = policy_name.strip().lower().replace("-", "_")

    if normalized_name in {"random", "random_policy", "random_allocation"}:
        return RandomPolicy(n_arms=n_arms, seed=seed)

    if normalized_name in {"ucb", "ucb1", "upper_confidence_bound"}:
        return UCBPolicy(n_arms=n_arms, seed=seed)

    raise ValueError(
        f"Unknown policy '{policy_name}'. Supported policies are: random, ucb."
    )
