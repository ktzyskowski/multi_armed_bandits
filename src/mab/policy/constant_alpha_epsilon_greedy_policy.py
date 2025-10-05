from typing import Optional

import numpy as np

from .policy import Policy


class ConstantAlphaEpsilonGreedyPolicy(Policy):
    """Constant alpha epsilon greedy policy."""

    def __init__(
        self,
        alpha: float,
        eps: float,
        k: int,
        random_seed: Optional[int] = None,
    ) -> None:
        """Initialize policy.

        Args:
            eps (float): epsilon parameter.
            k (int): number of actions to pick from.
            random_seed (int, optional): random seed for reproducibility.
        """
        if not (0 <= eps <= 1):
            raise ValueError("Invalid epsilon.")
        self._alpha = alpha
        self._eps = eps
        self._k = k
        self._rng = np.random.default_rng(seed=random_seed)
        self._q = np.zeros(k, dtype=np.float32)

    def __call__(self) -> int:
        if self._rng.random() < self._eps:
            # select a random action
            return self._rng.choice(self._k)
        else:
            # select the greedy action
            return np.argmax(self._q)  # type: ignore

    def update(self, action: int, reward: float):
        self._q[action] += self._alpha * (reward - self._q[action])
