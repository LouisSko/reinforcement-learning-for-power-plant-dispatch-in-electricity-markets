# This is the programmed Environment in which the power plant acts
import gym
from gym.spaces import Box, Discrete
import numpy as np
import random
import pandas as pd
import datetime
from datetime import timedelta

#get definitions from other files
from market import market_clearing

#define the market environment
class market_env(gym.Env):



    def __init__(self, states_list=[], demand=[], re = [], prices = [], eps_length=24,  capacity=200, mc=30, reward_scaling=1):

        """
            The customized initialisation of the environment.

            Returns:

        """

        #get predefined stuff
        super(market_env, self).__init__()

        #defineing different points in time of the environment
        self.time_list_hours = pd.to_datetime(states_list.values, utc=True).tolist()
        self.time_list_days= pd.DataFrame(filter(lambda d: (d.hour == 0), self.time_list_hours), columns=['Days'])
       

        #define variabels of environment from input data
        self.demand = demand
        self.re_gen = re
        self.prices = prices

        #factor by which the rewrad is scaled, so that we have a reward around 0 
        self.reward_scaling=reward_scaling

        #get data technical data for the agent
        self.capacity = capacity
        self.mc = mc

        #define start and end of learning
        self.Start = min(self.time_list_hours)
        self.End = max(self.time_list_hours)
        self.iter = 0
        self.eps_length= eps_length

        #set the first date that is analysed to the first in the data set
        self.date=self.Start

        # reset profit to 0
        self.profit = 0

        #Initialize results data frame
        self.results_ep = pd.DataFrame(columns=["reward", "market price", "bid price", "bid volume"])



        #define possible OBSERVATIONS:
        #each hour of a day is one state with one observation, which holds the forecast for that entire day
        #day-ahead load forecast (observation[0]) and renewable forecast for wind onshore (observation[1]), offshore (observation[2]) and solar (observation[3])
        # as well as the marginal costs observation[4], which are just fix at this point
        #Note: if no renewable plant is dispatched this information could be reduced by using the residual load

        self.observation_space = Box(low=np.array([0, 0,0,0,0]), high=np.array([1,1,1,1,30]),
                                     shape=(5,), dtype=np.float32)

        #define possible ACTIONS:
        # in the market the agent is currently only able to set the  price (action) at which it bids and the volume is fixed
        # these are defined relatively to the marginal costs 
        
        #TODO bigger action space, also choose volume
        #TODO ideal would be the submission of a bidding curve as in reality


        self.action_space = Discrete(10, 0)


    #function that sampels days from the data
    def _observe_state(self, date):

        """
                Method that puts together all the forecasted information of the market
                and presses it together in the form of the action space

                Returns:
                array including all forecasts for one time step.
        """

        self._demand = self.demand[date]
        self._sun = self.re_gen['Forecasted Solar [MWh]'].loc[date]
        self._wind_off = self.re_gen['Forecasted Wind Offshore [MWh]'].loc[date]
        self._wind_on = self.re_gen['Forecasted Wind Onshore [MWh]'].loc[date]


        return np.concatenate((self._demand, self._sun,
                               self._wind_off, self._wind_on, self.mc),
                              axis=None)



    def step (self, action):

        """
            Take a step in environment, which equals bidding in one hour.
            We take the profit given by the market and convert it into reward

            Returns:
            The current observation and reward, as wel as whether the state is terminal or not.
        """

        #define current state as seen forecasts
        observation = self._observe_state(self.date)

        
        #get bid from action
        bid_volume = self.capacity
        #the bid price is ralive to the marginal costs
        bid_price = action/10 * 2 * self.mc 

        
        profit, da_price = market_clearing(self, bid_price, bid_volume, self.date)

        #scale the reward
        reward = (profit/self.reward_scaling)

        #write results
        self.results_ep.loc[self.date] = [round(reward,4),da_price, bid_price, bid_volume]


        #check if terminal state and define the next day that is used

        
        
        if self.iter == self.eps_length-1:
            is_terminal = True

            #pick new random DATE (note not hour, we want our Rl agent ot look at each day)
            #the new state is then set to the first hour 
            i = random.randrange(len(self.time_list_days)-1)
            self.date=str(self.time_list_days.iloc[i]['Days'])
        
        else:
            is_terminal = False
            self.iter = self.iter + 1
            self.date=str(pd.to_datetime(self.date)+timedelta(hours = 1))

        
        #have little place holder for info as gym requires it
        info={}

        

        return observation, round(reward,4), is_terminal, info



    def reset (self):
        """
            The first method called before the experiment starts, as it resets the environment.
            Also it is called between different learning session, alias tryiing out new algorithms etc

            Returns:
            The first state from the environment.
        """

        #reset profit to 0
        self.profit=0
        self.iter =0

        #set state to be non-terminal
        self.is_terminal=False

        #reset date to randome new date
        i = random.randrange(len(self.time_list_days)-1)
        self.date=str(self.time_list_days.iloc[i]['Days'])


        return self._observe_state(str(self.date))

