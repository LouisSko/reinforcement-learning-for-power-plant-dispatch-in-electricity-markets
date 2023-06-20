import unittest
import pandas as pd
import numpy as np
from unittest.mock import Mock, MagicMock
from environment import market_env


class TestMarketEnv(unittest.TestCase):

    def setUp(self):
        demand = pd.DataFrame({'demand': [100, 200, 300]}, index=pd.to_datetime(['2020-06-20 00:00', '2020-06-20 01:00', '2020-06-20 02:00']))
        re = pd.DataFrame({'Forecasted Solar [MWh]': [10, 20, 30], 'Forecasted Wind Offshore [MWh]': [10, 20, 30], 'Forecasted Wind Onshore [MWh]': [10, 20, 30]}, index=pd.to_datetime(['2020-06-20 00:00', '2020-06-20 01:00', '2020-06-20 02:00']))
        capacity_forecast = pd.DataFrame({'capacity': [50, 60, 70]}, index=pd.to_datetime(['2020-06-20 00:00', '2020-06-20 01:00', '2023-06-20 02:00']))
        capacity_actual = pd.DataFrame({'capacity': [55, 65, 75]}, index=pd.to_datetime(['2020-06-20 00:00', '2020-06-20 01:00', '2023-06-20 02:00']))
        prices = pd.DataFrame({'price': [40, 50, 60]}, index=pd.to_datetime(['2020-06-20 00:00', '2020-06-20 01:00', '2020-06-20 02:00']))
        self.env = market_env(demand, re, capacity_forecast, capacity_actual, prices)

    def test_observe_state(self):
        date = pd.Timestamp('2020-06-20 02:00')
        state = self.env.observe_state(date)
        self.assertEqual(len(state), 16)
        print('test_observe_state passed')

    def test_rescale(self):
        profit = 5000
        lower_bound = -10000
        upper_bound = 10000
        reward = self.env.rescale(profit, lower_bound, upper_bound)
        self.assertTrue(-1 <= reward <= 1)

    def test_step(self):
        action = (Mock(item=MagicMock(return_value=5)), Mock(item=MagicMock(return_value=5)))
        TRAIN = True
        next_state, reward, is_terminal, truncated, info, avg_bid_price, bid_volume_list, capacity_current_list = self.env.step(action, TRAIN)
        self.assertTrue(-1 <= reward <= 1)

    def test_market_clearing(self):
        bid_price = 40
        bid_volume = 10
        actual_price = 50
        capacity_current = 15
        date = pd.Timestamp('2020-06-20 02:00')
        reward = self.env.market_clearing(bid_price, bid_volume, actual_price, capacity_current, date)
        self.assertTrue(-1 <= reward <= 1)

    def test_reset(self):
        TRAIN = True
        state, info = self.env.reset(TRAIN)
        self.assertEqual(len(state), 16)
        self.assertFalse(self.env.is_terminal)

    def test_get_random_new_date(self):
        TRAIN = True
        date = self.env.get_random_new_date(TRAIN)
        self.assertIsInstance(date, pd.Timestamp)

if __name__ == '__main__':
    unittest.main()
