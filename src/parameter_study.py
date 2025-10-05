from multiprocessing import Pool

import click
import numpy as np
import plotly.graph_objects as go

from mab.testbed import Testbed

# hard-coded experiment configuration for parameter study
CONFIG = {
    "n_arms": 10,
    "n_runs": 2000,
    "n_steps": 1000,
    "experiments": [
        # Epsilon greedy
        *[
            {
                "name": f"Epsilon Greedy (eps={eps})",
                "type": "eps_greedy",
                "stationary": True,
                "args": {"eps": eps},
            }
            for eps in [1 / 128, 1 / 64, 1 / 32, 1 / 16, 1 / 8, 1 / 4]
        ],
        # Gradient bandit
        *[
            {
                "name": f"Gradient (alpha={alpha})",
                "type": "gradient",
                "stationary": True,
                "args": {"alpha": alpha},
            }
            for alpha in [1 / 32, 1 / 16, 1 / 8, 1 / 4, 1 / 2, 1, 2, 4]
        ],
        # UCB
        *[
            {
                "name": f"Upper Confidence Bound (c={c})",
                "type": "ucb",
                "stationary": True,
                "args": {"c": c},
            }
            for c in [1 / 16, 1 / 8, 1 / 4, 1 / 2, 1, 2, 4]
        ],
        # Greedy w/ optimistic start
        *[
            {
                "name": f"Greedy w/ Optimistic Start (alpha=0.1, initial_q={initial_q})",
                "type": "eps_greedy",
                "stationary": True,
                "args": {"alpha": 0.1, "initial_q": initial_q, "eps": 0},
            }
            for initial_q in [1 / 2, 1, 2, 4]
        ],
    ],
}


@click.command()
@click.option("-p", "--n-processes", default=10)
def main(n_processes: int):
    k = CONFIG.get("n_arms")
    n_runs = CONFIG.get("n_runs")
    n_steps = CONFIG.get("n_steps")
    testbed = Testbed(k=k, n_runs=n_runs, n_steps=n_steps)

    experiments = CONFIG.get("experiments")
    with Pool(processes=n_processes) as pool:
        results = pool.map(testbed.run, experiments)

    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    # concatenate rewards from all experiments in 0th dim:
    # new dims = [n_experiments, n_runs, n_steps]
    rewards = np.stack([result["rewards"] for result in results], axis=0)

    # produce single scalar average reward per experiment configuration
    average_rewards = rewards.mean(axis=(1, 2))

    print(average_rewards)

    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    traces = [
        {
            "x": [1 / 128, 1 / 64, 1 / 32, 1 / 16, 1 / 8, 1 / 4],
            "y": average_rewards[0:6],
            "name": "Epsilon Greedy",
            "color": "red",
        },
        {
            "x": [1 / 32, 1 / 16, 1 / 8, 1 / 4, 1 / 2, 1, 2, 4],
            "y": average_rewards[6:14],
            "name": "Gradient Bandit",
            "color": "green",
        },
        {
            "x": [1 / 16, 1 / 8, 1 / 4, 1 / 2, 1, 2, 4],
            "y": average_rewards[14:21],
            "name": "Upper Confidence Bound",
            "color": "blue",
        },
        {
            "x": [1 / 2, 1, 2, 4],
            "y": average_rewards[21:25],
            "name": "Greedy w/ Optimistic Start",
            "color": "black",
        },
    ]

    # create the figure
    fig = go.Figure()

    # add traces for each method
    for trace in traces:
        fig.add_trace(
            go.Scatter(
                x=trace["x"],
                y=trace["y"],
                mode="lines",
                name=trace["name"],
                line={"color": trace["color"], "width": 2},
            )
        )

    fig.update_layout(
        title="Parameter Study",
        xaxis={"title": "Parameter", "type": "log"},
        yaxis={"title": "Average reward over first 1000 steps"},
    )
    fig.show()


if __name__ == "__main__":
    main()
