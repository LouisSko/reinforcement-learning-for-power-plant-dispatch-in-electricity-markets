import unittest
import torch
import numpy as np
from unittest.mock import Mock, patch
from environment import market_env
from ppo_torch import PPOAgent

class TestMainMethods(unittest.TestCase):
    
    def setUp(self):
        # Define any common resources needed for tests
        self.state_dim = 5
        self.price_action_dim = 3
        self.volume_action_dim = 3
        self.lr_actor = 0.0002
        self.lr_critic = 0.0008
        self.gamma = 0.99
        self.n_epochs = 10
        self.eps_clip = 0.22
        self.device = torch.device('cpu')

        self.ppo_agent = PPOAgent(self.state_dim, self.price_action_dim, self.volume_action_dim, 
                                  self.lr_actor, self.lr_critic, self.gamma, self.n_epochs, self.eps_clip, self.device)

        self.env = Mock()  # create a mock environment object
        self.env.observation_space.shape = [self.state_dim]
        self.env.action_space = [Mock(n=self.price_action_dim), Mock(n=self.volume_action_dim)]

    def test_select_action(self):
        # Test action selection
        state = torch.rand(self.state_dim)  # Random state
        action, _, _, _, _ = self.ppo_agent.select_action(state)
        self.assertIsInstance(action.item(), int)  # Convert tensor to integer and check

if __name__ == '__main__':
    unittest.main()
