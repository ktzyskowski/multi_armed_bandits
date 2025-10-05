from mab.bandit.stationary_bandit import MultiArmedBandit


class NonstationaryMultiArmedBandit(MultiArmedBandit):
    def step(self, action: int) -> float:
        reward = super().step(action)
        # simulate random walk on true q*(a) values
        self._q += self._rng.normal(loc=0.0, scale=0.01, size=self._q.shape)
        return reward
