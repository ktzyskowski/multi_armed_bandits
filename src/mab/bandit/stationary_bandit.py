from typing import Optional

import numpy as np


class MultiArmedBandit:
    """Stationary multi-armed bandit."""

    def __init__(
        self,
        k: int = 10,
        random_seed: Optional[int] = None,
    ) -> None:
        """Create a new multi-armed bandit.

        Args:
            k (int, optional): number of bandit arms.
            random_seed (int, optional): random seed for reproducibility.
        """
        self._rng = np.random.default_rng(seed=random_seed)
        # sample true value q*(a) for each action: N(0, 1)
        self._q = np.array([self._rng.normal(0.0, 1.0) for _ in range(k)])

    def actions(self) -> set[int]:
        return set(range(len(self._q)))

    def step(self, action: int) -> float:
        """Sample a reward from the multi-armed bandit.

        Args:
            action (int): index of the selected bandit arm.

        Raises:
            ValueError: if the selected bandit arm is invalid.

        Returns:
            float: sampled reward.
        """
        if action not in range(len(self._q)):
            raise ValueError("Invalid action.")
        # sample reward from selected bandit arm: N(q*(a), 1)
        reward = self._rng.normal(self._q[action], 1.0)
        return reward
