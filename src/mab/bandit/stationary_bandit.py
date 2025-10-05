from typing import Optional

import numpy as np


class MultiArmedBandit:
    """Stationary multi-armed bandit."""

    def __init__(
        self,
        k: int = 10,
        random_seed: Optional[int] = None,
        mean: float = 0.0,
    ) -> None:
        """Create a new multi-armed bandit.

        Args:
            k (int, optional): number of bandit arms.
            random_seed (int, optional): random seed for reproducibility.
            mean (float, optional): shift parameter for true rewards.
        """
        self._rng = np.random.default_rng(seed=random_seed)
        # sample true value q*(a) for each action: N(mean, 1)
        self._q = np.array([self._rng.normal(mean, 1.0) for _ in range(k)])

    def step(self, action: int) -> float:
        """Sample a reward from the multi-armed bandit.

        Args:
            action (int): index of the selected bandit arm.

        Raises:
            ValueError: if the selected bandit arm is invalid.

        Returns:
            float: sampled reward.
        """
        # sample reward from selected bandit arm: N(q*(a), 1)
        reward = self._rng.normal(self._q[action], 1.0)
        return reward
