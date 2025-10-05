from typing import Optional

import numpy as np

from .policy import Policy


def softmax(dist: np.ndarray):
    ans = np.exp(dist)
    ans /= np.sum(ans)
    return ans


class GradientPolicy(Policy):
    """Gradient policy."""

    def __init__(
        self,
        alpha: int,
        k: int,
        random_seed: Optional[int] = None,
        baseline: bool = True,
    ) -> None:
        self._alpha = alpha
        self._k = k
        self._rng = np.random.default_rng(seed=random_seed)
        self._h = np.zeros(k, dtype=np.float32)
        self._use_baseline = baseline
        self._baseline = 0
        self._t = 0

    def _pi(self) -> np.ndarray:
        pi = softmax(self._h)
        return pi

    def __call__(self) -> int:
        pi = self._pi()
        # sample from policy
        return self._rng.choice(np.arange(self._k), p=pi)

    def update(self, action: int, reward: float) -> None:
        pi = self._pi()
        self._t += 1

        if self._use_baseline:
            # update baseline, incremental sample-average
            # (keeping in line with Sutton and Barto, baseline R_bar(t) also includes R(t))
            self._baseline += (1 / self._t) * (reward - self._baseline)

        # update action preferences
        self._h = np.where(
            np.arange(self._k) == action,
            self._h + self._alpha * (reward - self._baseline) * (1 - pi),
            self._h - self._alpha * (reward - self._baseline) * pi,
        )
        # if self._use_baseline:
