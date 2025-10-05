from typing import Optional

import numpy as np

from .policy import Policy


class EpsilonGreedyPolicy(Policy):
    """Epsilon greedy policy."""

    def __init__(
        self,
        eps: float,
        k: int,
        random_seed: Optional[int] = None,
        initial_q: float = 0.0,
        alpha: Optional[float] = None,
    ) -> None:
        """Initialize policy.

        Args:
            eps (float): epsilon parameter.
            k (int): number of actions to pick from.
            random_seed (int, optional): random seed for reproducibility.
            initial_q (float): initial estimate of q*, useful for optimistic starts.
            alpha (float, optional): constant step-size update parameter.
        """
        if not (0 <= eps <= 1):
            raise ValueError("Invalid epsilon.")
        self._eps = eps
        self._k = k
        self._rng = np.random.default_rng(seed=random_seed)
        self._q = np.full(k, fill_value=initial_q, dtype=np.float32)
        self._n = np.zeros(k, dtype=np.int32)
        self._alpha = alpha

    def __call__(self) -> int:
        if self._rng.random() < self._eps:
            # select a random action
            return self._rng.choice(self._k)
        else:
            # select the greedy action
            return np.argmax(self._q)  # type: ignore

    def _step_size(self, action) -> float:
        if self._alpha:
            return self._alpha
        else:
            self._n[action] += 1
            return 1 / self._n[action]

    def update(self, action: int, reward: float) -> None:
        self._q[action] += self._step_size(action) * (reward - self._q[action])
