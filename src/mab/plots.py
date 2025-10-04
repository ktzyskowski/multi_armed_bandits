import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure


def plot_rewards(rewards: dict[str, np.ndarray]) -> Figure:
    fig, ax = plt.subplots()
    for key in rewards.keys():
        # average rewards across runs
        average_rewards = rewards[key].mean(axis=0)
        ax.plot(np.arange(len(average_rewards)), average_rewards, label=key)
        ax.set(
            xlabel="Step",
            ylabel="Average Reward",
        )
        ax.grid()
    ax.legend()
    return fig
