import json
from multiprocessing import Pool

import click
import numpy as np
import plotly.graph_objects as go

from mab.testbed import Testbed


@click.command()
@click.argument("filename")
@click.option("-p", "--n-processes", default=4)
def main(filename: str, n_processes: int):
    with open(filename, "r") as f:
        config = json.load(f)

    testbed = Testbed(
        k=config.get("n_arms"),
        n_runs=config.get("n_runs"),
        n_steps=config.get("n_steps"),
    )

    experiments = config.get("experiments")
    with Pool(processes=n_processes) as pool:
        results = pool.map(testbed.run, experiments)

    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    # concatenate rewards from all experiments in 0th dim:
    # new dims = [n_experiments, n_runs, n_steps]
    rewards = np.stack([result["rewards"] for result in results], axis=0)

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
