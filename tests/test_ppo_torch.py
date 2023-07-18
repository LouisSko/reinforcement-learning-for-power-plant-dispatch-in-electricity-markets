import pytest
import torch
import torch.nn as nn
from unittest.mock import MagicMock, patch
from ppo_torch import Buffer, ActorCritic, PPOAgent

# Test the Buffer class
def test_buffer():
    buffer = Buffer()

    # Test store_memory
    buffer.store_memory('state', 'price_action', 'volume_action', 'price_action_logprob', 'volume_action_logprob', 'state_value', 'reward', 'done')
    assert buffer.states == ['state']
    assert buffer.price_actions == ['price_action']
    assert buffer.volume_actions == ['volume_action']
    assert buffer.price_action_logprobs == ['price_action_logprob']
    assert buffer.volume_action_logprobs == ['volume_action_logprob']
    assert buffer.state_values == ['state_value']
    assert buffer.rewards == ['reward']
    assert buffer.dones == ['done']

    # Test clear
    buffer.clear()
    assert buffer.states == []
    assert buffer.price_actions == []
    assert buffer.volume_actions == []
    assert buffer.price_action_logprobs == []
    assert buffer.volume_action_logprobs == []
    assert buffer.state_values == []
    assert buffer.rewards == []
    assert buffer.dones == []

# Test the ActorCritic class
def test_actor_critic():
    actor_critic = ActorCritic(3, 3, 3)
    result = actor_critic.forward(torch.tensor([1., 2., 3.]))
    assert len(result) == 3
    assert isinstance(result[0], torch.distributions.Categorical)
    assert isinstance(result[1], torch.distributions.Categorical)
    assert result[2].shape == torch.Size([1])

# Test the PPOAgent class
def test_ppo_agent():
    ppo_agent = PPOAgent(3, 3, 3, 0.01, 0.01, 0.99, 10, 0.2, 32, "cpu")

    # Test send_memory_to_buffer with mock tensor values
    mock_state = torch.tensor([1.0, 2.0, 3.0])
    mock_price_action = torch.tensor(1)
    mock_volume_action = torch.tensor(2)
    mock_price_action_logprob = torch.tensor(1.0)
    mock_volume_action_logprob = torch.tensor(2.0)

    ppo_agent.send_memory_to_buffer(mock_state, mock_price_action, mock_volume_action, mock_price_action_logprob, mock_volume_action_logprob, 1.0, 2.0, True)

    assert ppo_agent.buffer.states == [mock_state]
    assert ppo_agent.buffer.price_actions == [mock_price_action]
    assert ppo_agent.buffer.volume_actions == [mock_volume_action]
    assert ppo_agent.buffer.price_action_logprobs == [mock_price_action_logprob]
    assert ppo_agent.buffer.volume_action_logprobs == [mock_volume_action_logprob]
    assert ppo_agent.buffer.state_values == [1.0]
    assert ppo_agent.buffer.rewards == [2.0]
    assert ppo_agent.buffer.dones == [True]

    # Test select_action
    result = ppo_agent.select_action(torch.tensor([1., 2., 3.]))
    assert len(result) == 5
    assert isinstance(result[0], torch.Tensor)
    assert isinstance(result[1], torch.Tensor)
    assert isinstance(result[2], torch.Tensor)
    assert isinstance(result[3], torch.Tensor)
    assert isinstance(result[4], torch.Tensor)

    # Test update with mock methods
    with patch.object(torch.distributions.Categorical, "log_prob", return_value=torch.tensor(1.0)), \
         patch.object(torch.optim.Adam, "zero_grad"), \
         patch.object(torch.optim.Adam, "step"), \
         patch.object(nn.Module, "load_state_dict"):
        ppo_agent.update()
        assert ppo_agent.buffer.states == []




