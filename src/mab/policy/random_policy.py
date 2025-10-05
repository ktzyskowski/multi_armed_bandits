from typing import Optional

import numpy as np

from .policy import Policy


class RandomPolicy(Policy):
    """Random policy.

    Each action is given equal probability of being selected.
    """

    def __init__(self, k: int, random_seed: Optional[int] = None) -> None:
        """Initialize policy.

        Args:
            k (int): number of actions to pick from.
            random_seed (int, optional): random seed for reproducibility.
        """
        self._k = k
        self._rng = np.random.default_rng(seed=random_seed)

    def __call__(self) -> int:
        return self._rng.choice(self._k)
