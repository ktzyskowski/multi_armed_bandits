from typing import Callable

import numpy as np
import tqdm

from mab.bandit import MultiArmedBandit
from mab.policy import Policy


class Simulator:
    def __init__(
        self,
        policy_factory: Callable[[int], Policy],
        bandit_factory: Callable[[int], MultiArmedBandit],
    ):
        self._policy_factory = policy_factory
        self._bandit_factory = bandit_factory

    def simulate(self, n_runs: int, n_steps: int):
        rewards = np.zeros((n_runs, n_steps), dtype=np.float32)
        for run_idx in tqdm.trange(n_runs, desc=f"Simulating {n_runs:,} bandit runs"):
            policy_random_seed = run_idx
            policy = self._policy_factory(policy_random_seed)
            bandit_random_seed = run_idx + n_runs + n_steps
            bandit = self._bandit_factory(bandit_random_seed)
            for step_idx in range(n_steps):
                action = policy()
                reward = bandit.step(action)
                rewards[run_idx, step_idx] = reward
        return rewards
