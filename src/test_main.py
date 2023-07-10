import unittest
import torch
import numpy as np
from unittest.mock import Mock, patch
from environment import market_env
from ppo_torch import PPOAgent
import pytest

@pytest.fixture
def common_resources():
    # Define any common resources needed for tests
    state_dim = 5
    price_action_dim = 3
    volume_action_dim = 3
    lr_actor = 0.0002
    lr_critic = 0.0008
    gamma = 0.99
    n_epochs = 10
    eps_clip = 0.22
    device = torch.device('cpu')
    ppo_agent = PPOAgent(state_dim, price_action_dim, volume_action_dim,
                         lr_actor, lr_critic, gamma, n_epochs, eps_clip, device)

    env = Mock()  # create a mock environment object
    env.observation_space.shape = [state_dim]
    env.action_space = [Mock(n=price_action_dim), Mock(n=volume_action_dim)]

    return ppo_agent, env, state_dim

def test_select_action(common_resources):
    ppo_agent, env, state_dim = common_resources
    # Test action selection
    state = torch.rand(state_dim)  # Random state
    action, _, _, _, _ = ppo_agent.select_action(state)
    assert isinstance(action, torch.Tensor), "action is not a tensor"
    assert isinstance(action.item(), int),  "action.item() is not an integer"
