from matplotlib import pyplot as plt

from mab.bandit import MultiArmedBandit
from mab.plots import plot_rewards
from mab.policy.random_policy import RandomPolicy
from mab.simulator import Simulator


def main():
    # create testbed simulator
    k = 10
    simulator = Simulator(
        policy_factory=lambda random_seed: RandomPolicy(
            k=k,
            random_seed=random_seed,
        ),
        bandit_factory=lambda random_seed: MultiArmedBandit(
            k=k,
            random_seed=random_seed,
        ),
    )

    # collect simulated rewards
    rewards = simulator.simulate(n_runs=2_000, n_steps=1_000)
    print(rewards.shape)

    # plot results!
    _fig = plot_rewards({"random": rewards})
    plt.show()


if __name__ == "__main__":
    main()
