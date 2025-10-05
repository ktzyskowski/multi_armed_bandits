import functools
from multiprocessing import Pool

import numpy as np
import plotly.graph_objects as go

from mab.bandit.nonstationary_bandit import NonstationaryMultiArmedBandit
from mab.bandit.stationary_bandit import MultiArmedBandit
from mab.policy.egreedy_policy import EpsilonGreedyPolicy
from mab.policy.gradient_policy import GradientPolicy
from mab.policy.random_policy import RandomPolicy
from mab.policy.ucb_policy import UpperConfidenceBoundPolicy
from mab.testbed import Testbed

N_ARMS = 10
N_RUNS = 2_000
N_STEPS = 1_000
N_PROCESSES = 4


def make_random_policy(random_seed: int):
    return RandomPolicy(k=N_ARMS, random_seed=None)


def make_epsilon_greedy_policy(random_seed: int, eps: float):
    return EpsilonGreedyPolicy(eps=eps, k=N_ARMS, random_seed=None)


def make_ucb_policy(random_seed: int, c: float):
    return UpperConfidenceBoundPolicy(c=c, k=N_ARMS, random_seed=None)


def make_gradient_policy(random_seed: int, alpha: float, baseline: bool):
    return GradientPolicy(alpha=alpha, baseline=baseline, k=N_ARMS, random_seed=None)


def run(testbed_config: dict):
    if testbed_config.get("nonstationary"):
        bandit_factory = lambda random_seed: NonstationaryMultiArmedBandit(
            k=N_ARMS,
            random_seed=None,
        )
    else:
        bandit_factory = lambda random_seed: MultiArmedBandit(
            k=N_ARMS,
            random_seed=None,
        )

    testbed = Testbed(
        policy_factory=testbed_config.get("policy_factory"),
        bandit_factory=bandit_factory,
    )
    rewards = testbed.run(N_RUNS, N_STEPS)
    return rewards


def main():
    testbed_configs = [
        # {
        #     "nonstationary": False,
        #     "name": "eps_greedy (eps=0.1)",
        #     "policy_factory": functools.partial(make_epsilon_greedy_policy, eps=0.1),
        # },
        # {
        #     "nonstationary": False,
        #     "name": "ucb (c=2)",
        #     "policy_factory": functools.partial(make_ucb_policy, c=2),
        # },
        {
            "nonstationary": False,
            "name": "gradient (alpha=0.1)",
            "policy_factory": functools.partial(
                make_gradient_policy, alpha=0.1, baseline=True
            ),
        },
        {
            "nonstationary": False,
            "name": "gradient (alpha=0.4)",
            "policy_factory": functools.partial(
                make_gradient_policy, alpha=0.4, baseline=True
            ),
        },
        {
            "nonstationary": False,
            "name": "gradient (alpha=0.1, no-baseline)",
            "policy_factory": functools.partial(
                make_gradient_policy, alpha=0.1, baseline=False
            ),
        },
        {
            "nonstationary": False,
            "name": "gradient (alpha=0.4, no-baseline)",
            "policy_factory": functools.partial(
                make_gradient_policy, alpha=0.4, baseline=False
            ),
        },
    ]

    with Pool(processes=N_PROCESSES) as pool:
        results = pool.map(run, testbed_configs)

    # concatenate rewards from all experiments in 0th dim:
    # new dims = [n_experiments, n_runs, n_steps]
    rewards = np.stack(results, axis=0)

    # average across runs
    average_rewards = rewards.mean(axis=1)
    # make plot of average reward over steps
    fig = go.Figure()
    fig.update_layout(xaxis_title="Step", yaxis_title="Average Reward")
    fig.add_hline(y=0.0, line_dash="dash")
    for i in range(len(testbed_configs)):
        fig.add_trace(
            go.Scatter(
                y=average_rewards[i],
                mode="lines",
                name=testbed_configs[i].get("name"),
            )
        )
    fig.show()


if __name__ == "__main__":
    main()
