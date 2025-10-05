from typing import Optional, TypedDict

import numpy as np
import tqdm

from mab.bandit.nonstationary_bandit import NonstationaryMultiArmedBandit
from mab.bandit.stationary_bandit import MultiArmedBandit
from mab.policy import Policy
from mab.policy.egreedy_policy import EpsilonGreedyPolicy
from mab.policy.gradient_policy import GradientPolicy
from mab.policy.random_policy import RandomPolicy
from mab.policy.ucb_policy import UpperConfidenceBoundPolicy


class TestbedConfig(TypedDict):
    name: str
    type: str
    stationary: bool
    args: Optional[dict[str, any]]
    bandit_args: Optional[dict[str, any]]


class Testbed:
    def __init__(
        self,
        k: int,
        n_runs: int,
        n_steps: int,
    ):
        self._k = k
        self._n_runs = n_runs
        self._n_steps = n_steps

    def policy_factory(
        self,
        config: TestbedConfig,
        random_seed: int = None,
    ) -> Policy:
        if config.get("type") == "random":
            return RandomPolicy(k=self._k, random_seed=random_seed)
        elif config.get("type") == "eps_greedy":
            return EpsilonGreedyPolicy(
                k=self._k, random_seed=random_seed, **config.get("args")
            )
        elif config.get("type") == "ucb":
            return UpperConfidenceBoundPolicy(
                k=self._k, random_seed=random_seed, **config.get("args")
            )
        elif config.get("type") == "gradient":
            return GradientPolicy(
                k=self._k, random_seed=random_seed, **config.get("args")
            )

    def bandit_factory(
        self,
        config: TestbedConfig,
        random_seed: int = None,
    ) -> MultiArmedBandit:
        bandit_args = config.get("bandit_args") or dict()
        if config.get("stationary"):
            return MultiArmedBandit(
                k=self._k,
                random_seed=random_seed,
                **bandit_args,
            )
        else:
            return NonstationaryMultiArmedBandit(
                k=self._k,
                random_seed=random_seed,
                **bandit_args,
            )

    def run(
        self,
        config: TestbedConfig,
        verbose: bool = False,
    ) -> dict[str, np.ndarray]:
        rewards = np.zeros((self._n_runs, self._n_steps), dtype=np.float32)
        was_optimal = np.zeros((self._n_runs, self._n_steps), dtype=np.float32)

        iterator = (
            tqdm.trange(self._n_runs, desc=f"Simulating bandit runs...")
            if verbose
            else range(self._n_runs)
        )

        for run_idx in iterator:
            policy_random_seed = run_idx
            policy = self.policy_factory(config, random_seed=policy_random_seed)
            bandit_random_seed = run_idx + self._n_runs + self._n_steps
            bandit = self.bandit_factory(config, random_seed=bandit_random_seed)
            for step_idx in range(self._n_steps):
                action = policy()
                # log if selected action is optimal
                was_optimal[run_idx, step_idx] = action == bandit.optimal_action()

                # take action, collect and log reward
                reward = bandit.step(action)
                rewards[run_idx, step_idx] = reward

                # update policy after observing (action, reward) sample
                policy.update(action, reward)

        return {"rewards": rewards, "was_optimal": was_optimal}
