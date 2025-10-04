from typing import Optional

import numpy as np


class RandomPolicy:
    def __init__(self, k: int, random_seed: Optional[int] = None) -> None:
        self._k = k
        self._rng = np.random.default_rng(seed=random_seed)

    def __call__(self) -> int:
        return self._rng.choice(self._k)
