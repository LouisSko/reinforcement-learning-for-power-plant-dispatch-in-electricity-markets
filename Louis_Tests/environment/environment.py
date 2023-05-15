import gym
from gym import spaces
import numpy as np
from typing import Optional, Union


class PowerPlantEnv(gym.Env):
    def __init__(self, input_data, prices):
        super().__init__()

        # Initialize your environment variables and parameters here
        self.input_data = input_data
        self.prices = prices
        self.marginal_costs = 30
        self.bid_volume = 1000
        self.current_day = 0
        self.hours = 24
        self.state = None
        self.reward_scaling = 10

        self.action_values = [0, np.inf]
        self.action_space = spaces.Discrete(len(self.action_values))

        self.observation_space = spaces.Box(low=0, high=np.inf, shape=(24,), dtype=np.float32)

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        # Reset the environment to its initial state
        self.current_day = 0
        self.state = self._get_observation()

        return np.array(self.state, dtype=np.float32)

    def step(self, action):
        # Perform one step in the environment given the action
        # calculate reward
        profit = 0

        for hour in range(self.hours):
            bid_price = self.action_values[action[hour]]
            bid_volume = self.bid_volume
            actual_price = self.prices.iloc[self.current_day, hour]
            profit += self.market_clearing(bid_price, bid_volume, actual_price)

        # scale the reward
        reward = (profit / self.reward_scaling)
        # increase the day by one
        self.current_day += 1
        # next observation
        self.state = self._get_observation()
        # check termination criterion
        terminated = self._check_termination()

        return np.array(self.state, dtype=np.float32), reward, terminated

    def render(self, mode='human'):
        # Render the environment (optional)
        ...

    def close(self):
        # Clean up resources (if any)
        ...

    def market_clearing(self, bid_price, bid_volume, actual_price):

        if bid_price <= actual_price:
            # bid is successful
            profit = bid_volume * (actual_price - self.marginal_costs)
        else:
            # bid is not successful
            profit = 0
        return profit

    def _get_observation(self):
        # Erstelle und gib die Beobachtung zurück, basierend auf dem aktuellen Zeitpunkt und anderen relevanten Informationen

        observation = self.input_data.iloc[self.current_day]

        return np.array(observation, dtype=np.float32)

    def _check_termination(self):
        # Überprüfe, ob das Environment terminiert ist
        # Hier ein einfacher Beispiel, bei dem das Environment für 10 Tage läuft und dann terminiert
        return self.current_day >= len(self.prices)-1
