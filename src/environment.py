"""
This is the programmed Environment in which the power plant acts
"""

# Importing necessary libraries for the power plant environment
import gymnasium as gym
from gymnasium.spaces import Box, Discrete, Tuple, MultiDiscrete
import numpy as np
import random
import pandas as pd

# define the market environment
class market_env(gym.Env):
    """
    This is a custom environment for a power plant market simulation where the power plant can act.
    It inherits the `gym.Env` class from OpenAI's gym module.

    The environment takes in demand, renewable energy, capacity forecasts, actual capacity and prices data. 
    It allows for actions in the form of bid price and bid volume in the market.

    The environment is episodic and returns a reward for every step based on the profit calculated using market prices and the power plant's bid.

    The observation space consists of past market data and the power plant's marginal costs. The action space consists of bid price and volume. 

    Parameters
    ----------
    demand : pd.DataFrame
        DataFrame containing demand data.
    re : pd.DataFrame
        DataFrame containing renewable energy generation data.
    capacity_forecast : pd.DataFrame
        DataFrame containing capacity forecast data.
    capacity_actual : pd.DataFrame
        DataFrame containing actual capacity data.
    prices : pd.DataFrame
        DataFrame containing market prices data.
    eps_length : int, optional
        Number of hours per episode, by default 24.
    capacity : int, optional
        Total capacity of the power plant, by default 200.
    mc : int, optional
        Marginal costs of the power plant, by default 30.
    lower_bound : int, optional
        Lower bound to rescale rewards, by default -10000.
    upper_bound : int, optional
        Upper bound to rescale rewards, by default 10000.
    """
    def __doc__(self):
        """
        Print a documentation string for this class
        """
        print("This is a custom environment for a power plant market simulation where the power plant can act.")

    # Initialization of the environment   
    def __init__(self, demand, re, capacity_forecast, capacity_actual, prices, eps_length=24, capacity=200, mc=30,
                 lower_bound=-10000, upper_bound=10000, train=True):
        """
        Initializes the environment with the given parameters.

        Args:
            demand: Pandas DataFrame of demand data.
            re: Pandas DataFrame of renewable energy generation data.
            capacity_forecast: Pandas DataFrame of capacity forecast data.
            capacity_actual: Pandas DataFrame of actual capacity data.
            prices: Pandas DataFrame of market prices data.
            eps_length: Number of hours for each episode. Default is 24 hours.
            capacity: The total capacity of the power plant.
            mc: Marginal costs of the power plant.
            lower_bound: Lower bound to rescale rewards. Default is -10000.
            upper_bound: Upper bound to rescale rewards. Default is 10000.
        """
        # All the necessary initializations and function calls go here.
        # get predefined stuff
        super().__init__()

        self.train = train

        # get rows where all data is available
        self.states_list = set(demand.index) & set(re.index) & set(capacity_forecast.index) & set(
            capacity_actual.index) & set(prices.index)
        # defining different points in time of the environment
        self.time_list_hours = pd.Series(list(self.states_list)).sort_values()
        self.time_list_days = pd.Series(filter(lambda d: (d.hour == 0), self.time_list_hours))

        # define lists to track actions in the tensorboard
        self.avg_bid_price = []
        self.capacity_current_list = []
        self.bid_volume_list = []
        self.profit_list = []
        self.profit_heuristic_list = []

        # initialize variables
        self.bid_price_heuristic = None
        self.bid_volume_heuristic = None
        self.reward = None
        self.bid_price = None
        self.profit_heuristic = None

        # set lower and upper bound to rescale rewards to -1 and 1
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound

        # define variables of environment from input data
        self.demand = demand
        self.re_gen = re
        self.capacity_forecast = capacity_forecast
        self.capacity_actual = capacity_actual
        self.prices = prices
        self.capacity_current = None

        # get data technical data for the agent
        self.capacity = capacity
        self.mc = mc

        # define start and end of learning
        self.Start = min(self.time_list_hours)
        self.End = max(self.time_list_hours)
        self.iter = 0
        self.eps_length = eps_length

        # set the first date that is analysed to the first in the data set
        self.date = self.Start
        self.observation = None
        # reset profit to 0
        self.profit = 0
        self.is_terminal = False
        # Initialize results data frame
        self.results_ep = pd.DataFrame(
            columns=["reward", "profit", "net_profit", "delta", "market price", "bid price", "bid volume",
                     "actual volume"])

        # define possible OBSERVATIONS:
        # Observation[0]: Day-ahead load forecast
        # Observation[1]: Renewable forecast for wind onshore
        # Observation[2]: Renewable forecast for wind offshore
        # Observation[3]: Renewable forecast for solar
        # Observation[4]: Marginal costs

        # expanded shape to incorporate expected forecast
        high = np.ones(15)
        last_element = np.array([self.mc / self.mc])
        high = np.concatenate((high, last_element))
        self.observation_space = Box(low=np.zeros(16), high=high,
                                     shape=(16,), dtype=np.float64)

        # define possible ACTIONS:
        # in the market the agent is currently only able to set the  price (action) at which it bids and the volume is fixed
        # these are defined relatively to the marginal costs 

        # TODO: bigger action space, also choose volume
        # TODO: ideal would be the submission of a bidding curve as in reality
        
        #self.action_space = Tuple((
        #    Discrete(50),  # Price action space (0 to 49)
        #    Discrete(50)  # Volume action space (0 to 9)
        #))

        self.action_space = MultiDiscrete((
            50,
            50
        ))


    # function that sampels days from the data
    def observe_state(self, date):

        """
                Method that puts together all the forecasted information of the market
                and presses it together in the form of the state space

                Args:
                    self.date (pd date): the random date 

                Returns:
                    array including all forecasts for the last three time steps.
        """

        # get also the last time steps from the last two hours
        time_range = pd.Timedelta(hours=2)

        # get necessary features x 3 
        self._demand = self.demand.loc[(self.demand.index >= date - time_range) & (self.demand.index <= date)]
        self._sun = self.re_gen['Forecasted Solar [MWh]'].loc[
            (self.re_gen['Forecasted Solar [MWh]'].index >= date - time_range) & (
                        self.re_gen['Forecasted Solar [MWh]'].index <= date)]
        self._wind_off = self.re_gen['Forecasted Wind Offshore [MWh]'].loc[
            (self.re_gen['Forecasted Wind Offshore [MWh]'].index >= date - time_range) & (
                        self.re_gen['Forecasted Wind Offshore [MWh]'].index <= date)]
        self._wind_on = self.re_gen['Forecasted Wind Onshore [MWh]'].loc[
            (self.re_gen['Forecasted Wind Onshore [MWh]'].index >= date - time_range) & (
                        self.re_gen['Forecasted Wind Onshore [MWh]'].index <= date)]
        self._capacity = self.capacity_forecast.loc[
            (self.capacity_forecast.index >= date - time_range) & (self.capacity_forecast.index <= date)]

        concat = np.concatenate((self._demand,
                                 self._sun,
                                 self._wind_off,
                                 self._wind_on,
                                 self._capacity,
                                 self.mc / self.mc), axis=None)  # why division by 10?

        # if the last time steps do not exist (e.g. end of the beginning of the day -> add zeros)
        if concat.size <= self.observation_space.shape[0]:
            concat = np.pad(concat, (16 - concat.size, 0), 'constant', constant_values=0)

        return concat

    def rescale_linear(self, profit, lower_bound, upper_bound):
        """
        Rescales the given profit to a reward in the range of -1 to 1.

        Args:
            profit (float): The profit to be rescaled.
            lower_bound (float): The lower bound of the profit.
            upper_bound (float): The upper bound of the profit.

        Returns:
            float: The rescaled profit.
        """
        reward = (profit - lower_bound) / (upper_bound - lower_bound) * 2 - 1
        return reward

    def rescale_linearV2(self, profit, lower_bound, upper_bound, bid_price, mc):
        """
        Rescales the given profit to a reward in the range of -1 to 1.
        This method also takes the bid price and marginal costs into account.

        Args:
            profit (float): The profit to be rescaled.
            lower_bound (float): The lower bound of the profit.
            upper_bound (float): The upper bound of the profit.
            bid_price (float): The price at which the power plant bid in the market.
            mc (float): The marginal cost.

        Returns:
            float: The rescaled profit.
        """
        profit -= abs(mc-bid_price)
        reward = (profit - lower_bound) / (upper_bound - lower_bound) * 2 - 1
        return reward

    def rescale_log(self, profit, lower_bound, upper_bound):
        """
        Rescales the given profit to a reward using a logarithmic scale.

        Args:
            profit (float): The profit to be rescaled.
            lower_bound (float): The lower bound of the profit.
            upper_bound (float): The upper bound of the profit.

        Returns:
            float: The rescaled profit.
        """

        #handle the options of positive & negative profit
        if profit > 0:
            reward = np.log(profit + 1).item()
        elif profit < 0:
            reward = -np.log(-profit + 1).item()
        else:
            reward = 0

        return reward

    def step(self, action):
        """
        Takes a step in the environment, simulating bidding in the market for one hour.
        It returns the new state of the environment, the reward for the step,
        whether the step led to a terminal state, and additional info.

        Args:
            action (tuple): The action to take in the form of (bid price, bid volume).
            TRAIN (bool): Whether the environment is being used for training or not.

        Returns:
            tuple: A tuple containing the new state, the reward, 
                whether the state is terminal or not, whether the episode was truncated, and additional info.
        """

        # get bids from action
        action_price = action[0].item()
        action_volume = action[1].item()

        # the bid price is relative to the marginal costs
        # bid_price = (action_price / 2 * self.mc)

        # increase the possible bid volume range
        # bid_volume = action_volume * 20

        # set bid_price and bid_volume using an exponential function
        self.bid_price = np.power(action_price, 2) * self.mc / (25 * 25)
        self.bid_volume = np.power(action_volume, 2) * self.capacity / (100 * 25)

        # use a heuristic which always uses mc as bid price and the forecasted load as bid volume
        self.bid_price_heuristic = self.mc
        self.bid_volume_heuristic = self.capacity_forecast.loc[self.date].values[0] * self.capacity

        # current capacity and forecast of pv
        self.capacity_current = self.capacity_actual.loc[self.date].values[0] * self.capacity

        # market price
        self.actual_price = self.prices.loc[self.date].values[0]

        # list of bid volumes and current capacities to plot it in the tensorboard later
        self.bid_volume_list.append(self.bid_volume)
        self.capacity_current_list.append(self.capacity_current)

        # we only track the average bid price when the bid price can have an impact on the profit
        if self.capacity_current > 0:
            self.avg_bid_price.append(self.bid_price)

        # calculate the profit
        profit = self.market_clearing(self.bid_price, self.bid_volume, self.actual_price)
        profit_heuristic = self.market_clearing(self.bid_price_heuristic, self.bid_volume_heuristic, self.actual_price)

        # add profit to list
        self.profit_list.append(profit)
        self.profit_heuristic_list.append(profit_heuristic)

        # calculate reward -> scaled profit
        self.reward = self.rescale_linearV2(profit, self.lower_bound, self.upper_bound, self.bid_price, self.mc)
        # self.reward = self.rescale_linear(profit, self.lower_bound, self.upper_bound)
        # self.reward = self.rescale_log(profit,  self.lower_bound, self.upper_bound)

        # print('bid price: ', bid_price, ' actual price: ', actual_price, 'bid volume: ', bid_volume, ' current: ',
        #      int(self.capacity_current), 'reward: ', reward, 'date: ', self.date)

        # check if terminal state and define the next day that is used
        if self.iter == self.eps_length - 1:
            self.is_terminal = True
            self.date = self.get_random_new_date()

        else:
            self.is_terminal = False
            self.iter = self.iter + 1
            self.date = self.date + pd.Timedelta(hours=1)

        # now update the state
        next_state = self.observe_state(self.date)

        # have little place holder for info and truncated as gym requires it
        info = {}
        truncated = False
        return next_state, round(self.reward, 4), self.is_terminal, truncated, info

    def market_clearing(self, bid_price, bid_volume, actual_price):
        """
        Simulates the market clearing given a bid price and volume, and the actual market price.
        It returns the overall profit from the market in EUR and the realized market price in EUR/MWh.

        Args:
            bid_price (float): The price at which the power plant is bidding.
            bid_volume (float): The volume that the power plant is bidding.
            actual_price (float): The actual price in the market.

        Returns:
            float: The profit from the market.
        """
        if bid_price <= actual_price:
            # bid is successful
            profit = bid_volume * (actual_price - self.mc)

            if bid_volume > int(self.capacity_current):
                # agent has to buy at the market for a higher price. Simplified assumptions
                if actual_price > 0:
                    avg_price = actual_price * 1.3
                if actual_price < 0:
                    avg_price = actual_price * 0.7
                if actual_price == 0:
                    avg_price = 3

                # calculate the excess volume
                remaining = bid_volume - int(self.capacity_current)

                # update the reward
                profit = profit - remaining * (actual_price - self.mc)  # substract the excess reward from the reward which should not be granted
                profit = profit + remaining * (actual_price - avg_price)  # add the costs of buying additional energy. avg_price is essentially the new mc's

                # its equivalent: reward = reward - remaining * (avg_price - self.mc)

        else:
            # bid is not successful
            profit = 0

        return profit


    def reset(self, **kwargs):
        """
        Resets the environment to its initial state.

        Args:
            TRAIN (bool): Whether the environment is being used for training or not.

        Returns:
            tuple: A tuple containing the initial state of the environment and an empty dictionary.
        """
        # reset profit to 0
        self.profit = 0
        self.iter = 0
        # set state to be non-terminal
        self.is_terminal = False
        # reset date to randome new date
        self.date = self.get_random_new_date()

        return self.observe_state(self.date), {}

    def get_random_new_date(self):
        """
        Pick a random date and time from the available dataset.
        If in training mode, the new state is set to the first hour of a random day.
        For testing, a random day in July 2021 is selected.

        Args:
            TRAIN (boolean): Flag to determine if the agent is in training or testing mode.

        Returns:
            The randomly selected date and time.
        """


        # If in training mode, pick a new random day.
        if self.train:

            random_index = random.randrange(len(self.time_list_days) - 1)

            # Select the random day.
            self.date = self.time_list_days.iloc[random_index]

            # If the selected day is in September 2021, recursively select another day.
            # This leaves data from September 2021 for testing.
            if self.date.strftime('%m') == '09' and self.date.strftime('%Y') == '2021':
                self.get_random_new_date()

        # If in testing mode, pick a random day in July 2021.
        else:
            random_day = random.randrange(1, 28)
            self.date = pd.Timestamp('2021-07-{day} 00:00:00+02:00'.format(day=str(random_day)))

        return self.date
