import functools
from multiprocessing import Pool

import numpy as np
import plotly.graph_objects as go

from mab.bandit.nonstationary_bandit import NonstationaryMultiArmedBandit
from mab.bandit.stationary_bandit import MultiArmedBandit
from mab.policy.epsilon_greedy_policy import EpsilonGreedyPolicy
from mab.policy.random_policy import RandomPolicy
from mab.testbed import Testbed

N_ARMS = 10
N_RUNS = 2_000
N_STEPS = 1_000


def make_random_policy(random_seed: int):
    return RandomPolicy(k=N_ARMS, random_seed=random_seed)


def make_epsilon_greedy_policy(random_seed: int, eps: float):
    return EpsilonGreedyPolicy(eps=eps, k=N_ARMS, random_seed=random_seed)


def run(testbed_config: dict):
    if testbed_config.get("nonstationary"):
        bandit_factory = lambda random_seed: NonstationaryMultiArmedBandit(
            k=N_ARMS,
            random_seed=random_seed,
        )
    else:
        bandit_factory = lambda random_seed: MultiArmedBandit(
            k=N_ARMS,
            random_seed=random_seed,
        )

    testbed = Testbed(
        policy_factory=testbed_config.get("policy_factory"),
        bandit_factory=bandit_factory,
    )
    rewards = testbed.run(N_RUNS, N_STEPS)
    return rewards


def main():
    testbed_configs = [
        {
            "nonstationary": False,
            "name": "random",
            "policy_factory": make_random_policy,
        },
        {
            "nonstationary": True,
            "name": "eps_greedy (eps=0.1, nonstationary)",
            "policy_factory": functools.partial(make_epsilon_greedy_policy, eps=0.1),
        },
        {
            "nonstationary": False,
            "name": "eps_greedy (eps=0.1, stationary)",
            "policy_factory": functools.partial(make_epsilon_greedy_policy, eps=0.1),
        },
    ]

    n_processes = 4
    with Pool(processes=n_processes) as pool:
        results = pool.map(run, testbed_configs)

    rewards = np.stack(results, axis=0)
    average_rewards = rewards.mean(axis=1)

    # create plots
    fig = go.Figure()
    fig.update_layout(xaxis_title="Step", yaxis_title="Average Reward")

    for i in range(len(testbed_configs)):
        fig.add_trace(
            go.Scatter(
                y=average_rewards[i], mode="lines", name=testbed_configs[i].get("name")
            )
        )
    fig.show()


if __name__ == "__main__":
    main()
