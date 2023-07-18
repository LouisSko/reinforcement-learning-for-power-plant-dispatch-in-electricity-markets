import torch
from unittest.mock import patch, call
from hp_tuning import hp_tuning

# Mocking rl_agent_run to avoid real training during testing
@patch('hp_tuning.rl_agent_run')
def test_hp_tuning(mock_rl_agent_run):
    hp_tuning()  # Call the function

    # Get the arguments of the first call to the mock
    args, kwargs = mock_rl_agent_run.call_args

    # Extract the dictionary
    hp_dict = args[0]

    # Define the expected keys
    expected_keys = {
        'lower_bound',
        'upper_bound',
        'batch_size',
        'n_episodes',
        'update_timestep',
        'n_epochs',
        'eps_clip',
        'gamma',
        'lr_actor',
        'lr_critic'
    }

    # Assert that the keys in hp_dict are the same as the expected keys
    assert set(hp_dict.keys()) == expected_keys

