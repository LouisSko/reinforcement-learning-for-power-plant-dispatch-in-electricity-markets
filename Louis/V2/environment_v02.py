# This is the programmed Environment in which the power plant acts
import gym
from gym.spaces import Box, Discrete, Tuple
import numpy as np
import random
import pandas as pd
import datetime
import torch


# define the market environment
class market_env(gym.Env):

    def __init__(self, demand, re, capacity_forecast, capacity_actual, prices, eps_length=24, capacity=200, mc=30):
        """

        """
        # get predefined stuff
        super().__init__()

        # get rows where all data is available
        self.states_list = set(demand.index) & set(re.index) & set(capacity_forecast.index) & set(capacity_actual.index) & set(prices.index)
        # defining different points in time of the environment
        self.time_list_hours = pd.Series(list(self.states_list)).sort_values()
        self.time_list_days = pd.Series(filter(lambda d: (d.hour == 0), self.time_list_hours))

        # define variables of environment from input data
        self.demand = demand
        self.re_gen = re
        self.capacity_forecast = capacity_forecast
        self.capacity_actual = capacity_actual
        self.prices = prices

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
        self.results_ep = pd.DataFrame(columns=["reward", "market price", "bid price", "bid volume"])

        # define possible OBSERVATIONS:
        # Observation[0]: Day-ahead load forecast
        # Observation[1]: Renewable forecast for wind onshore
        # Observation[2]: Renewable forecast for wind offshore
        # Observation[3]: Renewable forecast for solar
        # Observation[4]: Marginal costs

        # expanded shape to incorporate expected forecast
        self.observation_space = Box(low=np.array([0, 0, 0, 0, 0, 0]), high=np.array([1, 1, 1, 1, 1, 30]),
                                     shape=(6,))

        # define possible ACTIONS:
        # in the market the agent is currently only able to set the  price (action) at which it bids and the volume is fixed
        # these are defined relatively to the marginal costs 

        # TODO: bigger action space, also choose volume
        # TODO: ideal would be the submission of a bidding curve as in reality

        self.action_space = Tuple((
            Discrete(50),  # Price action space (0 to 49)
            Discrete(10)  # Volume action space (0 to 9)
        ))

    # function that sampels days from the data
    def observe_state(self, date):

        """
                Method that puts together all the forecasted information of the market
                and presses it together in the form of the action space

                Returns:
                array including all forecasts for one time step.
        """

        self._demand = self.demand.loc[date]
        self._sun = self.re_gen['Forecasted Solar [MWh]'].loc[date]
        self._wind_off = self.re_gen['Forecasted Wind Offshore [MWh]'].loc[date]
        self._wind_on = self.re_gen['Forecasted Wind Onshore [MWh]'].loc[date]
        self._capacity = self.capacity_forecast.loc[date]

        return np.concatenate((self._demand,
                               self._sun,
                               self._wind_off,
                               self._wind_on,
                               self._capacity,
                               self.mc), axis=None)

    def step(self, action):

        """
            Take a step in environment, which equals bidding in one hour.
            We take the profit given by the market and convert it into reward

            Returns:
            The current observation and reward, as well as whether the state is terminal or not.
        """

        # define current state as seen forecasts
        self.observation = self.observe_state(self.date)

        # get bids from action
        action_price = action[0]
        action_volume = action[1]

        # the bid price is relative to the marginal costs
        bid_price = action_price / 10 * self.mc
        bid_volume = action[1]*self.capacity/(self.action_space[1].n-1)

        # TODO: implement bid_volume restriction

        profit, da_price = self.market_clearing(bid_price, bid_volume, self.date)
        avg_price = da_price*1.2

        # define the reward
        # handling rewards close to zero can be problematic.-> add a constant of 1
        if profit > 0 and bid_volume <= self.capacity_actual:
            reward = np.log(profit)
        elif profit > 0 and bid_volume > self.capacity_actual:
            reward = np.log(profit - avg_price*(bid_volume-self.capacity_actual))  
        elif profit < 0 and bid_volume <= self.capacity_actual:
            reward = -np.log(-profit)
        elif profit < 0 and bid_volume > self.capacity_actual:
            reward = -np.log(-profit + avg_price*(bid_volume-self.capacity_actual))
        else:
            reward = 0

        # write results
        self.results_ep.loc[self.date] = [round(reward, 4), da_price, bid_price, bid_volume]

        # check if terminal state and define the next day that is used
        if self.iter == self.eps_length - 1:
            self.is_terminal = True
            self.date = self.get_random_new_date()

        else:
            self.is_terminal = False
            self.iter = self.iter + 1
            self.date = self.date + pd.Timedelta(hours=1)

        # have little place holder for info as gym requires it

        info = {}

        return self.observation, round(reward, 4), self.is_terminal, info

    def reset(self):
        """
            The first method called before the experiment starts, as it resets the environment.
            Also it is called between different learning session, alias tryiing out new algorithms etc

            Returns:
            The first state from the environment.
        """
        # reset profit to 0
        self.profit = 0
        self.iter = 0
        # set state to be non-terminal
        self.is_terminal = False
        # reset date to randome new date
        self.date = self.get_random_new_date()

        return self.observe_state(self.date)

    def get_random_new_date(self):
        # pick new random DATE (note not hour, we want our Rl agent to look at each day)
        # the new state is then set to the first hour
        i = random.randrange(len(self.time_list_days) - 1)
        self.date = self.time_list_days.iloc[i]

        return self.date

    def market_clearing(self, bid_price, bid_volume, date):
        """
            A function that calculates the output the day-ahead market would give when the selcted bid is submitted [EUR]

            Return: overall profit received from market in EUR and realised market price in EUR/MWh
        """

        # calculate revenue
        actual_price = self.prices.loc[date].values[0]

        if bid_price <= actual_price:
            # bid is sucessful
            profit = bid_volume * (actual_price - self.mc)

        else:
            # bid is not sucessfull
            profit = 0

        return profit, actual_price
