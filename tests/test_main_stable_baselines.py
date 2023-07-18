import os
import pytest
from unittest.mock import Mock, patch
from main_stable_baselines import train_a2c
import numpy as np

@patch('os.makedirs')
@patch('os.path.exists')
@patch('main_stable_baselines.A2C')
@patch('main_stable_baselines.check_env')
@patch('main_stable_baselines.market_env')
@patch('main_stable_baselines.read_processed_files')
def test_train_a2c(mock_read, mock_market_env, mock_check_env, mock_A2C, mock_exists, mock_makedirs):

    # Create mock return values for the read_processed_files function
    mock_read.return_value = (Mock(), Mock(), Mock(), Mock(), Mock(), Mock(), Mock(), Mock(), Mock())

    # Create a mock environment
    env_mock = Mock()
    env_mock.reset.return_value = np.ones(16), {}  # Add this line
    mock_market_env.return_value = env_mock

    # Create a mock model
    model_mock = Mock()
    model_mock.predict.return_value = np.array(1), {}
    mock_A2C.return_value = model_mock

    # Assume directories exist
    mock_exists.return_value = False

    # Test training case
    train_a2c(train=True)

    # Check directory creation calls
    #mock_exists.assert_any_call()
    #mock_exists.assert_any_call()

    # Check other function calls
    mock_read.assert_called_once()
    mock_market_env.assert_called_once()
    mock_check_env.assert_called_once_with(env_mock)
    mock_A2C.assert_called_once()
    model_mock.learn.assert_called_once()
    model_mock.save.assert_called_once()
