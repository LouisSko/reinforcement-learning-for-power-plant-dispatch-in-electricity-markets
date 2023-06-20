# This is the programmed Environment in which the power plant acts
import gymnasium as gym
from gymnasium.spaces import Box, Discrete, Tuple
import numpy as np
import random
import pandas as pd


# define the market environment
class market_env(gym.Env):

    def __init__(self, demand, re, capacity_forecast, capacity_actual, prices, eps_length=24, capacity=200, mc=30, lower_bound=-10000, upper_bound=10000):
        """

        """
        # get predefined stuff
        super().__init__()

        # get rows where all data is available
        self.states_list = set(demand.index) & set(re.index) & set(capacity_forecast.index) & set(capacity_actual.index) & set(prices.index)
        # defining different points in time of the environment
        self.time_list_hours = pd.Series(list(self.states_list)).sort_values()
        self.time_list_days = pd.Series(filter(lambda d: (d.hour == 0), self.time_list_hours))

        # define lists to track actions in the tensorboard
        self.avg_bid_price = []
        self.capacity_current_list = []
        self.bid_volume_list = []

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
        self.results_ep = pd.DataFrame(columns=["reward",  "profit", "net_profit", "delta", "market price", "bid price", "bid volume", "actual volume"])

        # define possible OBSERVATIONS:
        # Observation[0]: Day-ahead load forecast
        # Observation[1]: Renewable forecast for wind onshore
        # Observation[2]: Renewable forecast for wind offshore
        # Observation[3]: Renewable forecast for solar
        # Observation[4]: Marginal costs

        # expanded shape to incorporate expected forecast
        high = np.ones(15)
        last_element = np.array([self.mc/10])
        high = np.concatenate((high, last_element))
        self.observation_space = Box(low=np.zeros(16), high=high,
                                     shape=(16,))

        # define possible ACTIONS:
        # in the market the agent is currently only able to set the  price (action) at which it bids and the volume is fixed
        # these are defined relatively to the marginal costs 

        # TODO: bigger action space, also choose volume
        # TODO: ideal would be the submission of a bidding curve as in reality

        self.action_space = Tuple((
            Discrete(11),  # Price action space (0 to 49)
            Discrete(11)  # Volume action space (0 to 9)
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
        self._sun = self.re_gen['Forecasted Solar [MWh]'].loc[(self.re_gen['Forecasted Solar [MWh]'].index >= date - time_range) & (self.re_gen['Forecasted Solar [MWh]'].index <= date)]
        self._wind_off = self.re_gen['Forecasted Wind Offshore [MWh]'].loc[(self.re_gen['Forecasted Wind Offshore [MWh]'].index >= date - time_range) & (self.re_gen['Forecasted Wind Offshore [MWh]'].index <= date)]
        self._wind_on = self.re_gen['Forecasted Wind Onshore [MWh]'].loc[(self.re_gen['Forecasted Wind Onshore [MWh]'].index >= date - time_range) & (self.re_gen['Forecasted Wind Onshore [MWh]'].index <= date)]
        self._capacity = self.capacity_forecast.loc[(self.capacity_forecast.index >= date - time_range) & (self.capacity_forecast.index <= date)]
        
        
        concat = np.concatenate((self._demand,
                               self._sun,
                               self._wind_off,
                               self._wind_on,
                               self._capacity,
                               self.mc/10), axis=None)
        
        # if the last time steps do not exist (e.g. end of the beginning of the day -> add zeros)
        if concat.size <= self.observation_space.shape[0]:
            concat = np.pad(concat, (16 - concat.size, 0), 'constant', constant_values=0)
       
        return concat
    

    def rescale(self, profit, lower_bound, upper_bound):
            if profit > upper_bound:
                profit = upper_bound
            elif profit < lower_bound:
                profit = lower_bound
            
            reward = (profit - lower_bound) / (upper_bound - lower_bound) * 2 - 1
            return reward

    def step(self, action, TRAIN):

        """
            Take a step in environment, which equals bidding in one hour.
            We take the profit given by the market and convert it into reward

            Returns:
            The current observation and reward, as well as whether the state is terminal or not.
        """

        # should be at the end of step
        # self.observation = self.observe_state(self.date)


        # get bids from action
        action_price = action[0].item()
        action_volume = action[1].item()

        print('Aktion für den Zeitpunkt {date}'.format(date=self.date))
        print('Evaluation für den Zeitpunkt {date}'.format(date=self.date))

        # the bid price is relative to the marginal costs
        bid_price = (action_price / 5 * self.mc)

        # times 2 to increase the possible bid volume range
        bid_volume = action_volume  * 2

        # current capacity of pv 
        self.capacity_current = self.capacity_actual.loc[self.date].values[0]*self.capacity

        # market price
        actual_price = self.prices.loc[self.date].values[0]

        # for training and the reward function it's better to only allow positive prices
        if actual_price < 0:
            actual_price = 0

        # list of bid volumes and current capacities to plot it in the tensorboard later
        self.bid_volume_list.append(bid_volume)
        self.capacity_current_list.append(self.capacity_current)

        # we only track the average bid price when the bid price can have an impact on the profit
        if self.capacity_current > 0:
            self.avg_bid_price.append(bid_price)
        
        # calculate the profit / reward
        reward = self.market_clearing(bid_price, bid_volume, actual_price, self.capacity_current, self.date)


        #print('bid price: ', bid_price, ' actual price: ', actual_price, 'bid volume: ' , bid_volume, ' current: ' , int(self.capacity_current))
        
        self.bid_volume = bid_volume
        self.bid_price = bid_price
        self.da_price = actual_price
        self.reward = reward

        # check if terminal state and define the next day that is used
        if self.iter == self.eps_length - 1:
            self.is_terminal = True
            self.date = self.get_random_new_date(TRAIN)

        else:
            self.is_terminal = False
            self.iter = self.iter + 1
            self.date = self.date + pd.Timedelta(hours=1)

        # now update the state
        next_state = self.observe_state(self.date)
        print('State Update für den Zeitpunkt {date}'.format(date=self.date))

        # have little place holder for info and truncated as gym requires it
        info = {}
        truncated = False
        return next_state, round(reward, 4), self.is_terminal, truncated, info, self.avg_bid_price, self.bid_volume_list, self.capacity_current_list
    
    def market_clearing(self, bid_price, bid_volume, actual_price, capacity_current, date):
        """
        A function that calculates the output the day-ahead market would give when the selected bid is submitted [EUR]

        Returns: overall profit received from the market in EUR and realized market price in EUR/MWh
        """
        if capacity_current is None:
            # Handle the case when capacity_current is None
            return 0  # Default reward when capacity_current is None

        if bid_volume is not None and bid_volume > capacity_current:
            # agent has to buy at the market for a higher price
            avg_price = actual_price * 1.1
            remaining = bid_volume - capacity_current
            profit = remaining * (avg_price + self.mc)
        else:
            # bid is successful or bid_volume is None
            profit = bid_volume * (actual_price - self.mc)

        reward = self.rescale(profit, self.lower_bound, self.upper_bound)

        return reward


    def reset(self, TRAIN):
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
        self.date = self.get_random_new_date(TRAIN)

        return self.observe_state(self.date), {}

    def get_random_new_date(self, TRAIN):
        # pick new random DATE (note not hour, we want our Rl agent to look at each day)
        # the new state is then set to the first hour
        if TRAIN:
            valid_dates = self.time_list_days.tolist()
            if len(valid_dates) > 0:
                i = random.randrange(len(valid_dates))
                self.date = valid_dates[i]
            else:
                # Handle the case when no valid dates are found
                self.date = pd.Timestamp('2023-06-20 00:00:00')
        else:
            # For testing, generate a random date within the desired range
            day = random.randrange(1, 28)
            self.date = pd.Timestamp('2020-09-{day} 00:00:00+02:00'.format(day=str(day)))

        return self.date


    

