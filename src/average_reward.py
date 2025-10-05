from multiprocessing import Pool

import numpy as np
import plotly.graph_objects as go

from mab.testbed import Testbed, TestbedConfig

N_ARMS = 10
N_RUNS = 2_000
N_STEPS = 1_000
N_PROCESSES = 4


def main():
    experiments: list[TestbedConfig] = [
        {
            "name": "eps_greedy (eps=0.1)",
            "type": "eps_greedy",
            "stationary": True,
            "args": {"eps": 0.1},
        },
        {
            "name": "ucb (c=2)",
            "type": "ucb",
            "stationary": True,
            "args": {"c": 2.0},
        },
        {
            "name": "gradient (alpha=0.1)",
            "type": "gradient",
            "stationary": True,
            "args": {"alpha": 0.1},
        },
        {
            "name": "gradient (alpha=0.4)",
            "type": "gradient",
            "stationary": True,
            "args": {"alpha": 0.4},
        },
        {
            "name": "gradient (alpha=0.1, no-baseline)",
            "type": "gradient",
            "stationary": True,
            "args": {"alpha": 0.1, "baseline": True},
        },
        {
            "name": "gradient (alpha=0.4, no-baseline)",
            "type": "gradient",
            "stationary": True,
            "args": {"alpha": 0.4, "baseline": True},
        },
    ]

    testbed = Testbed(k=N_ARMS, n_runs=N_RUNS, n_steps=N_STEPS)
    with Pool(processes=N_PROCESSES) as pool:
        results = pool.map(testbed.run, experiments)

    # concatenate rewards from all experiments in 0th dim:
    # new dims = [n_experiments, n_runs, n_steps]
    rewards = np.stack(results, axis=0)

    # average across runs
    average_rewards = rewards.mean(axis=1)
    # make plot of average reward over steps
    fig = go.Figure()
    fig.update_layout(xaxis_title="Step", yaxis_title="Average Reward")
    fig.add_hline(y=0.0, line_dash="dash")
    for i in range(len(experiments)):
        fig.add_trace(
            go.Scatter(
                y=average_rewards[i],
                mode="lines",
                name=experiments[i].get("name"),
            )
        )
    fig.show()


if __name__ == "__main__":
    main()
