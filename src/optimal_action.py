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

    k = config.get("n_arms")
    n_runs = config.get("n_runs")
    n_steps = config.get("n_steps")
    testbed = Testbed(k=k, n_runs=n_runs, n_steps=n_steps)

    experiments = config.get("experiments")
    with Pool(processes=n_processes) as pool:
        results = pool.map(testbed.run, experiments)

    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    # concatenate rewards from all experiments in 0th dim:
    # new dims = [n_experiments, n_runs, n_steps]
    was_optimal = np.stack([result["was_optimal"] for result in results], axis=0) * 100

    # average across runs
    average_optimal_selected = was_optimal.mean(axis=1)
    # make plot of average reward over steps
    fig = go.Figure()
    fig.update_layout(xaxis_title="Step", yaxis_title="% Optimal Action Selected")
    fig.update_layout(xaxis={"range": [0, n_steps]}, yaxis={"range": [0, 100]})
    for i in range(len(experiments)):
        fig.add_trace(
            go.Scatter(
                y=average_optimal_selected[i],
                mode="lines",
                name=experiments[i].get("name"),
            )
        )
    fig.show()


if __name__ == "__main__":
    main()
