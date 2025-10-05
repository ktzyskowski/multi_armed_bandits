from typing import Callable

import numpy as np
import tqdm

from mab.bandit.stationary_bandit import MultiArmedBandit
from mab.policy import Policy


class Testbed:
    def __init__(
        self,
        policy_factory: Callable[[int], Policy],
        bandit_factory: Callable[[int], MultiArmedBandit],
    ):
        self._policy_factory = policy_factory
        self._bandit_factory = bandit_factory

    def run(self, n_runs: int, n_steps: int, verbose: bool = False):
        rewards = np.zeros((n_runs, n_steps), dtype=np.float32)

        iterator = (
            tqdm.trange(n_runs, desc=f"Simulating {n_runs:,} bandit runs")
            if verbose
            else range(n_runs)
        )

        for run_idx in iterator:
            policy_random_seed = run_idx
            policy = self._policy_factory(random_seed=policy_random_seed)
            bandit_random_seed = run_idx + n_runs + n_steps
            bandit = self._bandit_factory(random_seed=bandit_random_seed)
            for step_idx in range(n_steps):
                action = policy()
                reward = bandit.step(action)
                policy.update(action, reward)
                rewards[run_idx, step_idx] = reward
        return rewards
