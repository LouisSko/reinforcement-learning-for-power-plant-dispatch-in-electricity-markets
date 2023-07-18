from main_ppo import rl_agent_run
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock, mock_open
import pytest
import os

@pytest.fixture
def setup_hp_dict():
    return {
        'lower_bound': -20000,
        'upper_bound': 20000,
        'batch_size': 16,
        'n_episodes': 1,
        'update_timestep': 2048,
        'n_epochs': 10,
        'eps_clip': 0.22,
        'gamma': 0.99,
        'lr_actor': 0.0002,
        'lr_critic': 0.0008
    }


@patch('os.makedirs')
@patch('os.path.exists', return_value=False)
@patch('torch.utils.tensorboard.SummaryWriter')
@patch('pickle.dump')
@patch('main_ppo.PPOAgent')
@patch('main_ppo.market_env')
@patch('main_ppo.read_processed_files')
@patch("builtins.open", new_callable=mock_open)

# very difficult to write a meaningfull unit test for this function (is it even possible?)
def test_rl_agent_run(mock_open_file, mock_read_processed_files, mock_market_env, mock_PPOAgent, mock_pickle_dump,
                      mock_summary_writer, mock_exists, mock_makedirs, setup_hp_dict):
    # Set up mock objects
    mock_agent = MagicMock()
    mock_PPOAgent.return_value = mock_agent
    mock_agent.select_action.return_value = [0, 0, 0, 0, 0]

    mock_env = MagicMock()
    mock_market_env.return_value = mock_env
    mock_env.reset.return_value = [0, 0]
    mock_env.step.return_value = [0, 0, 'done', 0, {}]
    mock_env.observation_space.shape = [5]
    mock_env.action_space = [MagicMock(n=3), MagicMock(n=3)]

    dummy_df = pd.DataFrame(np.random.randint(0, 100, size=(100, 4)), columns=list('ABCD'))
    mock_read_processed_files.return_value = (
    dummy_df, dummy_df, dummy_df, dummy_df, dummy_df, dummy_df, dummy_df, dummy_df, dummy_df)

    # Mock the SummaryWriter methods that write to disk
    mock_writer = MagicMock()
    mock_summary_writer.return_value = mock_writer
    mock_writer.add_scalar = MagicMock()
    mock_writer.add_scalars = MagicMock()
    mock_writer.close = MagicMock()

    # Call the function
    rl_agent_run(setup_hp_dict, device='cpu', train=True, model_checkpoint=None, tb_name='my_experiment', logging=False)

    # Assert the PPOAgent's select_action are called
    assert mock_agent.select_action.called

