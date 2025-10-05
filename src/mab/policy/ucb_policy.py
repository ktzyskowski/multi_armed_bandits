from typing import Optional

import numpy as np

from .policy import Policy


class UpperConfidenceBoundPolicy(Policy):
    """Upper confidence bound policy."""

    def __init__(
        self,
        c: float,
        k: int,
        random_seed: Optional[int] = None,
    ) -> None:
        """Initialize policy.

        Args:
            c (float): exploration parameter.
            k (int): number of actions to pick from.
            random_seed (int, optional): random seed for reproducibility.
        """
        self._c = c
        self._k = k
        self._rng = np.random.default_rng(seed=random_seed)
        self._q = np.zeros(k, dtype=np.float32)
        self._n = np.zeros(k, dtype=np.int32)
        self._t = 0

    def __call__(self) -> int:
        # any action that has not been visited before is considered a maximizing action
        unvisited_indices = np.nonzero(self._n == 0)[0]
        if len(unvisited_indices) != 0:
            # arbitrarily pick first action where N(a) is 0
            action = unvisited_indices[0]
            return action

        # if all have been visited, use Q value + UCB term
        action = np.argmax(self._q + self._c * np.sqrt(np.log(self._t) / self._n))
        return action

    def update(self, action: int, reward: float):
        self._t += 1
        self._n[action] += 1
        self._q[action] += (1 / self._n[action]) * (reward - self._q[action])
