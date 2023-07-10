import unittest
import pandas as pd
import numpy as np
from unittest.mock import Mock, MagicMock
from environment import market_env
import pytest
from unittest.mock import MagicMock, create_autospec
import torch

@pytest.fixture
def market_env_fixture():
    # Arrange
    date = pd.Timestamp(year=2021, month=1, day=1, hour=2)
    date_range = pd.date_range(start='2021-01-01 00:00', end='2021-01-01 02:00', freq='H')

    # Create the demand DataFrame
    demand = pd.DataFrame({'demand': [0.3, 0.2, 0.9]}, index=date_range)

    # Create the re DataFrame
    re = pd.DataFrame({'Forecasted Solar [MWh]': [0.9, 0, 0],
                       'Forecasted Wind Offshore [MWh]': [0.4, 0.2, 1],
                       'Forecasted Wind Onshore [MWh]': [0.8, 0.2, 0.4]}, index=date_range)

    # Generate random data for the capacity_forecast DataFrame
    capacity_forecast = pd.DataFrame({'capacity': [0.1, 0.2, 0.4]}, index=date_range)

    # Generate random data for the capacity_actual DataFrame
    capacity_actual = pd.DataFrame({'capacity': [0.2, 0.1, 0.4]}, index=date_range)

    # Generate random data for the prices DataFrame
    prices = pd.DataFrame({'price': [-10, 230, 100]}, index=date_range)

    env = market_env(demand, re, capacity_forecast, capacity_actual, prices)
    env.capacity_current = 200
    env.date = date
    env.mc = 30


    return env


def test_observe_state(market_env_fixture):
    date = pd.Timestamp(year=2021, month=1, day=1, hour=2)

    # Construct a instance of market_env_fixture and its variables
    env = market_env_fixture

    # Pre-calculate expected output
    expected_output = np.array([0.3, 0.2, 0.9, 0.9, 0, 0, 0.4, 0.2, 1, 0.8, 0.2, 0.4, 0.1, 0.2, 0.4, 1])

    # observe state
    result = env.observe_state(date)

    # Assert
    assert isinstance(result, np.ndarray), "Result should be a numpy array"
    np.testing.assert_array_equal(result, expected_output), "The rescale function does not return the expected value."

def test_rescale_linear(market_env_fixture):

    env = market_env_fixture
    profit = 50
    lower_bound = 0
    upper_bound = 100

    # Expected output
    expected_output = (profit - lower_bound) / (upper_bound - lower_bound) * 2 - 1  # this should be 0

    # Act
    result = env.rescale_linear(profit, lower_bound, upper_bound)

    # Assert
    assert result == expected_output, "The rescale function does not return the expected value."

def test_rescale_linearV2(market_env_fixture):
    # Arrange
    env = market_env_fixture
    profit = 50.0
    lower_bound = 0.0
    upper_bound = 100.0
    bid_price = 10.0

    # Expected output
    expected_output = ((profit - abs(env.mc - bid_price)) - lower_bound) / (upper_bound - lower_bound) * 2 - 1  # this should compute to -0.4

    # Act
    result = env.rescale_linearV2(profit, lower_bound, upper_bound, bid_price, env.mc)

    # Assert
    assert result == expected_output, "The rescale function does not return the expected value."


def test_rescale_log(market_env_fixture):
    # Arrange
    env = market_env_fixture
    profit = 50.0
    lower_bound = 0.0
    upper_bound = 100.0

    # Expected output
    expected_output = np.log(profit + 1).item()  # this should compute to approximately 3.9318

    # Act
    result = env.rescale_log(profit, lower_bound, upper_bound)

    # Assert
    assert pytest.approx(result) == expected_output, "The rescale function does not return the expected value."

    # Test with negative profit
    profit = -50.0
    expected_output = -np.log(-profit + 1).item()  # this should compute to approximately -3.9318
    result = env.rescale_log(profit, lower_bound, upper_bound)
    assert pytest.approx(result) == expected_output, "The rescale function does not return the expected value when profit is negative."

    # Test with zero profit
    profit = 0.0
    expected_output = 0.0
    result = env.rescale_log(profit, lower_bound, upper_bound)
    assert result == expected_output, "The rescale function does not return the expected value when profit is zero."

def test_step(market_env_fixture):
    # Arrange
    env = market_env_fixture
    action = torch.tensor([5, 4])
    TRAIN = True

    # Act
    next_state, reward, is_terminal, truncated, info = env.step(action, TRAIN)

    # Assert
    assert isinstance(next_state, np.ndarray)
    assert next_state.shape[0] == 16
    assert isinstance(reward, np.float64) or isinstance(reward, float)
    assert isinstance(reward, np.float64) or isinstance(reward, float)
    assert is_terminal is False

def test_market_clearing(market_env_fixture):
    # Arrange
    env = market_env_fixture
    env.mc = 5
    env.capacity_current = 10
    bid_price = 8
    bid_volume = 12
    actual_price = 10

    # Act
    profit = env.market_clearing(bid_price, bid_volume, actual_price)

    # Assert
    expected_profit = (10 - 5) * 10 + (12 - 10) * (10 - 10 * 1.3)
    assert profit == expected_profit

    # Now test case when bid is not successful
    bid_price = 12
    profit = env.market_clearing(bid_price, bid_volume, actual_price)

    # Assert
    assert profit == 0

def test_reset(market_env_fixture):
    # Arrange
    env = market_env_fixture
    TRAIN = True
    env.profit = 100  # Set initial properties to non-default values
    env.iter = 5
    env.is_terminal = True
    env.time_list_days = pd.Series(pd.date_range(start='2021-01-01 00:00', end='2021-05-01 02:00', freq='d'))

    # Act
    state, info = env.reset(TRAIN)

    # Assert
    assert env.profit == 0
    assert env.iter == 0
    assert env.is_terminal is False
    assert len(state) == 16  # Assuming state is a numpy array with size 16 as per previous context.
    assert info == {}

def test_get_random_new_date(market_env_fixture):

    env = market_env_fixture

    env.time_list_days = pd.Series(pd.date_range(start='2021-01-01 00:00', end='2021-05-01 02:00', freq='d'))

    # Check for training mode
    random_date_train = env.get_random_new_date(TRAIN=True)
    assert isinstance(random_date_train, pd.Timestamp)
    assert random_date_train.strftime('%m') != '09' or random_date_train.strftime('%Y') != '2021'

    # Check for testing mode
    random_date_test = env.get_random_new_date(TRAIN=False)
    assert isinstance(random_date_test, pd.Timestamp)
    assert random_date_test.strftime('%m') == '07' and random_date_test.strftime('%Y') == '2021'

