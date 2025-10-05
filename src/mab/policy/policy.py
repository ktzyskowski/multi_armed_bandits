from abc import ABC, abstractmethod


class Policy(ABC):
    """Base policy interface."""

    @abstractmethod
    def __call__(self) -> int:
        """Sample from the policy.

        Returns:
            int: the selected action.
        """
        # must be implemented
        raise NotImplementedError()

    def update(self, action: int, reward: float) -> None:
        """Update the policy, after observing the given action and reward.

        Args:
            action (int): the selected action.
            reward (float): the received reward.
        """
        # no default implementation
        pass
